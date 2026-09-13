"""Visible indexing status; also keeps automatic catalogue polling subscribed."""
from homeassistant.components.sensor import SensorEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([HomeGuideLibrary(entry.runtime_data, entry.entry_id)])


class HomeGuideLibrary(CoordinatorEntity, SensorEntity):
    _attr_name = 'HomeGuide library'
    _attr_icon = 'mdi:bookshelf'
    _attr_native_unit_of_measurement = 'appliances'

    def __init__(self, coordinator, entry_id):
        super().__init__(coordinator)
        self._attr_unique_id = f'{entry_id}_library'

    @property
    def native_value(self):
        return len(self.coordinator.data['catalog']['appliances'])

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data
        return {'catalog_revision': data['catalog']['revision'],
                'appliances': data['catalog']['appliances'],
                'documents': [{k: d.get(k) for k in ('id', 'title', 'status', 'active', 'error')}
                              for d in data['documents']]}
