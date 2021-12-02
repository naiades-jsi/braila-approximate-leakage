
class NaNSensorsException(Exception):
    def __init__(self, sensors):
        self.message = "Missing values in the following sensors: {}".format(sensors)
        self.sensors_list = sensors

    def __str__(self):
        return repr(self.message)