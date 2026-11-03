"""Camera entity for the SwitchBot K10+ map."""
from __future__ import annotations

import logging

import aiohttp
from homeassistant.components.camera import Camera
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DEVICE_TYPE_K10, DEVICE_TYPE_TO_MODEL, DOMAIN
from .coordinator import SwitchBotS10Coordinator
from .k10_api import K10ProtocolError
from .map_renderer import render_map_png

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the map camera for supported devices."""
    if entry.data.get("device_type") != DEVICE_TYPE_K10:
        return

    coordinator: SwitchBotS10Coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([SwitchBotVacuumMapCamera(hass, coordinator)])


class SwitchBotVacuumMapCamera(Camera):
    """Render the current K10+ cloud map as a Home Assistant camera."""

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

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return the latest K10+ map image."""
        try:
            session = async_get_clientsession(self.hass)
            api = await self.coordinator._authorize_k10_api(session)
            map_data, _ = await api.get_map(session)
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
            _LOGGER.warning("K10+ map image refresh failed (%s)", type(exc).__name__)
            return self._last_image
