import numpy as np
from mag_manip import mag_manip

class SUPIEE:
    """
    用于构建SUPIEE对象的类
    :param name: SUPIEE对象的名称 (str)
    :param calibration_path: eMNS校准文件的路径 (str)
    """
    def __init__(self,
                 name: str="supiee",
                 calibration_path: str="../calib/mpem_calibration_file_sp=40_order=1.yaml") -> None:
        self.name = name
        self.calibration_path = calibration_path
        self.forward_model = mag_manip.ForwardModelMPEM()
        self.forward_model.setCalibrationFile(calibration_path)

    def currents_to_field(self,
                          currents: np.ndarray=np.array([0., 0., 0.]),
                          position: np.ndarray=np.array([0., 0., 0.])) -> np.ndarray:
        """
        应用前向模型计算给定位置的磁场。
        """
        bg_jac = self.forward_model.getFieldActuationMatrix(position)
        field = bg_jac.dot(currents)  # B = A * I
        return field

    def field_to_currents(self,
                          field: np.ndarray=np.array([0., 0., 0.]),
                          position: np.ndarray=np.array([0., 0., 0.])) -> np.ndarray:
        """
        应用反向模型计算在给定位置产生磁场所需的电流。
        """
        bg_jac = self.forward_model.getFieldActuationMatrix(position)
        currents = np.linalg.inv(bg_jac).dot(field)  # I = A^-1 * B
        return currents