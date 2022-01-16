
class NaNSensorsException(Exception):
    def __init__(self, sensors, epoch_timestamp):
        self.message = "Missing values in the following sensors: {}".format(sensors)
        self.sensors_list = sensors
        self.epoch_timestamp = epoch_timestamp

    def __str__(self):
        return repr(self.message)