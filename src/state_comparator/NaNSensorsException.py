
class NaNSensorsException(Exception):
    def __init__(self, sensors_name_alt_value_list, epoch_timestamp):
        """
        Creates error and constructs an informative error message.

        :param sensors_name_alt_value_list: List of tuple of 3 values, first value is the name of the sensors which
        has the wrong or missing values, the second is the alternative name of the sensors which is used on kafka and in
        this application to map it to the EPANET network. The third value is the float value of the sensor pressure.
        :param epoch_timestamp: UNIX timestamp in seconds.
        """
        self.sensor_list = [i[0] for i in sensors_name_alt_value_list]
        self.sensors_name_alt_value_list = sensors_name_alt_value_list
        self.epoch_timestamp = epoch_timestamp

        self.error_msg = f"Missing values in the following sensors at time {self.epoch_timestamp}:"
        for sensor_name, alternative_name, value in self.sensors_name_alt_value_list:
            self.error_msg += f" - name: {sensor_name}, alternative name: {alternative_name}, value: {value}"

    def __str__(self):
        return repr(self.error_msg)