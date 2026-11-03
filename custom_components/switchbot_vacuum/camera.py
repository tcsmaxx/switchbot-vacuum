"""Camera entity for the SwitchBot K10+/K10+ Pro map."""
from __future__ import annotations

import logging

import aiohttp
from homeassistant.components.camera import Camera
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DEVICE_TYPE_K10, DEVICE_TYPE_K10PRO, DEVICE_TYPE_TO_MODEL, DOMAIN
from .coordinator import SwitchBotS10Coordinator
from .k10_api import K10Api, K10ProtocolError
from .map_renderer import extract_route_points, render_map_png

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the map camera for supported K10-family devices."""
    if entry.data.get("device_type") not in (DEVICE_TYPE_K10, DEVICE_TYPE_K10PRO):
        return

    coordinator: SwitchBotS10Coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([SwitchBotVacuumMapCamera(hass, coordinator)])


class SwitchBotVacuumMapCamera(Camera):
    """Render the current K10-family cloud map as a Home Assistant camera."""

    _attr_has_entity_name = True
    _attr_name = "Map"
    _attr_icon = "mdi:map"
    _attr_content_type = "image/png"

    def __init__(self, hass: HomeAssistant, coordinator: SwitchBotS10Coordinator) -> None:
        super().__init__()
        self.hass = hass
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.device_mac}_map"
        device_type = coordinator.entry.data.get("device_type", DEVICE_TYPE_K10)
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.device_mac)},
            name=coordinator.device_name or "SwitchBot Vacuum",
            manufacturer="SwitchBot",
            model=DEVICE_TYPE_TO_MODEL.get(device_type, device_type),
        )
        self._last_image: bytes | None = None
        self._map_api: K10Api | None = None
        self._group_id: str | None = None
        self._last_route_diagnostic: tuple[str, int, int] | None = None

    async def _authorize_map_api(self, session: aiohttp.ClientSession) -> K10Api:
        """Authorize the delegated map API for K10+ and K10+ Pro.

        Upstream currently enables this Qihoo/Botslab API only for the K10+.
        The K10+ Pro is intentionally tried here in isolation so a failed map
        request cannot affect the normal Pro coordinator/control path.
        """
        await self.coordinator._ensure_token()
        if not self.coordinator.device_mac:
            raise K10ProtocolError("Vacuum device identifier is unavailable")

        if not self._group_id:
            devices = await self.coordinator.async_discover_devices()
            selected = next(
                (d for d in devices if d["device_mac"] == self.coordinator.device_mac),
                None,
            )
            if not selected or not selected.get("group_id"):
                raise K10ProtocolError("The selected vacuum has no family identifier")
            self._group_id = selected["group_id"]

        if self._map_api is None:
            self._map_api = K10Api(
                self.coordinator.device_mac,
                await self.coordinator._get_product_key(),
                self.coordinator._uuid.replace("-", ""),
            )

        if not self.coordinator.access_token or not self.coordinator.wonderlab_endpoint:
            raise K10ProtocolError("SwitchBot account session is unavailable")

        await self._map_api.authorize(
            session,
            self.coordinator.wonderlab_endpoint,
            self.coordinator.access_token,
            self._group_id,
        )
        return self._map_api

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return the latest K10-family map image."""
        try:
            session = async_get_clientsession(self.hass)
            api = await self._authorize_map_api(session)
            map_data, _ = await api.get_map(session)
            # Log metadata only: no raw payload, room geometry, tokens or signed URLs.
            payload = map_data.get("data", map_data) if isinstance(map_data, dict) else {}
            route = payload.get("posArray") if isinstance(payload, dict) else None
            route_kind = type(route).__name__
            route_size = len(route) if isinstance(route, (list, dict, str)) else 0
            decoded_count = len(extract_route_points(map_data))
            diagnostic = (route_kind, route_size, decoded_count)
            if diagnostic != self._last_route_diagnostic:
                _LOGGER.info(
                    "K10-family map route diagnostic: posArray_type=%s, "
                    "posArray_length=%d, parsed_points=%d",
                    *diagnostic,
                )
                self._last_route_diagnostic = diagnostic
            self._last_image = await self.hass.async_add_executor_job(
                render_map_png, map_data
            )
            return self._last_image
        except (
            aiohttp.ClientError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            K10ProtocolError,
        ) as exc:
            _LOGGER.warning(
                "K10-family map image refresh failed (%s)", type(exc).__name__
            )
            return self._last_image
