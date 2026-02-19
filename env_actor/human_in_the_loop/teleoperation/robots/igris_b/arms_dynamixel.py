import os
import sys
import time
import numpy as np
from dynamixel_sdk import *
import threading

from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
class maf:
    def __init__(self, maf_num=10,dim=12):
        self.predata=np.zeros([maf_num-1,dim])


    def filter(self, x):
        new=np.concatenate((self.predata,np.array([x])),axis=0)
        self.predata=new[1:]
        return list(new.mean(0))


class LowPassFilter1D:
    def __init__(self, fc, dt, init_val=0.0):
        RC = 1 / (2 * np.pi * fc)
        self.alpha = dt / (RC + dt)
        self.y = init_val

    def reset(self, init_val=0.0):
        self.y = init_val

    def filter(self, x):
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

    def filter_array(self, data):
        data = np.asarray(data)
        output = np.zeros_like(data, dtype=float)

        for i, x in enumerate(data):
            output[i] = self.filter(x)

        return output

HZ=25
UNITVEL2DEG = 0.229*360/60 
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi

PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

ADDR_PRESENT_POSITION = 132
LEN_GROUP_SYNC_READ = 4

ADDR_ENABLE_TORQUE = 64
LEN_ENABLE_TORQUE = 1

ADDR_LED = 65
LEN_LED = 1

ADDR_GOAL_POS = 116
LEN_GOAL_POS = 4

ADDR_GOAL_CUR = 102
LEN_GOAL_CUR = 2
class DxlMasterArm(Node):
    def __init__(self, use_part:list[str], baudrate=BAUDRATE):
        super().__init__('dxl_master_arm')
        self.declare_parameter('port', '/dev/igrisb_masterarm')
        self.port_name = self.get_parameter('port').value

        self.get_logger().info(f"Initializing DxlMasterArm with port {self.port_name} and baudrate {baudrate}")
        
        self.available_part_name=["right_arm","left_arm"]
        
        self.use_part = [name for name in use_part if name in self.available_part_name]
        
        self.ids={
            "right_arm":[11,12,13,14,15,16],
            "left_arm":[21,22,23,24,25,26],
        }
        
        self.q={
            "right_arm":np.zeros(6),
            "left_arm":np.zeros(6),            
        }
        
        self.baudrate = baudrate
        
        self.pos = []
        
        self.declare_parameter('joint_min', [-180.0, -20.0, -90.0, -130.0, -90.0, -30.0, -60.0, -160.0, -140.0, 0.0, -90.0, -30.0])
        self.declare_parameter('joint_max', [60.0, 160.0, 140.0, 0.0, 90.0, 40.0, 180.0, 20.0, 90.0, 130.0, 90.0, 40.0])

        self.joint_min_rad = list(map(lambda x: x * np.pi / 180, self.get_parameter('joint_min').get_parameter_value().double_array_value))
        self.joint_max_rad = list(map(lambda x: x * np.pi / 180, self.get_parameter('joint_max').get_parameter_value().double_array_value))

        self.portHandler = PortHandler(self.port_name)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_name}")
        
        if not self.portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")
        
        self.get_logger().info(f"Successfully opened port {self.port_name} with baudrate {baudrate}")
    
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_GROUP_SYNC_READ)
        self.groupSyncWriteTorque = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_ENABLE_TORQUE, LEN_ENABLE_TORQUE)
        self.groupSyncWriteLed = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_LED, LEN_LED)
        self.groupSyncWriteGoalPos = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POS, LEN_GOAL_POS)
        self.groupSyncWriteGoalCur = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_CUR, LEN_GOAL_CUR)
    
        self.groupSyncRead.clearParam()
        self.groupSyncWriteTorque.clearParam()
        self.groupSyncWriteLed.clearParam()

        self.lpf=maf(10,12)
                
        for id in self.all_ids():
            result = True
            result &= self.groupSyncRead.addParam(id)
            
            if not result:
                raise RuntimeError(f"Failed to add parameter for id {id}")

        self.sync_write_led(1,self.all_ids())
        self.sync_write_torque(0,self.all_ids())
                
        # self.joint_target_pub = self.create_publisher(JointState, '/igris_b/target_joints', 100)
        
        self.timer=self.create_timer(1/HZ,self.sync_read_position) 
    
    @staticmethod
    def normalize(x, interval:tuple):
        a, b = interval
        if a == b:
            raise ValueError("interval's start and end cannot be the same")
        norm = (x - a) / (b - a)
        return max(0.0, min(1.0, norm))  # 0 ~ 1 range clipping
    
    def all_ids(self):
        return self.select_ids(self.available_part_name)
    
    def select_ids(self,names:list[str]):
        return [id for name in self.use_part for id in self.ids[name] if name in names]
             
    def sync_read_position(self):
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to sync read position: {dxl_comm_result}")
            raise RuntimeError(f"Failed to sync read position: {dxl_comm_result}")
        
        data_available = True
        
        for name in self.use_part:
            for i, id in enumerate(self.ids[name]):
                data_available &= self.groupSyncRead.isAvailable(id, ADDR_PRESENT_POSITION, LEN_GROUP_SYNC_READ)
                if not data_available:
                    break
                position = self.dxl_to_radian(self.uint32_to_int32(self.groupSyncRead.getData(id, ADDR_PRESENT_POSITION, LEN_GROUP_SYNC_READ)))

                if name == "right_arm":
                    self.q[name][i] = np.clip(position, self.joint_min_rad[i], self.joint_max_rad[i])
                elif name == "left_arm":
                    self.q[name][i] = np.clip(position, self.joint_min_rad[i+6], self.joint_max_rad[i+6])
                else:
                    self.q[name][i] = position
        
        
        # joint_msg = JointState()
        # joint_msg.header.stamp = self.get_clock().now().to_msg()
        # joint_msg.name = [f"r_joint_{i}" for i in range(1, 7)] + [f"l_joint_{i}" for i in range(1, 7)]
        # joint_msg.position = list(self.q['right_arm'])+list(self.q['left_arm'])
        # self.joint_target_pub.publish(joint_msg)
        
    def sync_write_led(self, status,ids):
        data = self.divide_to_bytes(status, LEN_LED)
        self.groupSyncWriteLed.clearParam()
        for id in ids:
            result = self.groupSyncWriteLed.addParam(id, data)
            if not result:
                self.get_logger().error(f"Failed to add parameter for id {id}")
                raise RuntimeError(f"Failed to add parameter for id {id}")
        dxl_comm_result = self.groupSyncWriteLed.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to sync write led: {dxl_comm_result}")
            raise RuntimeError(f"Failed to sync write led: {dxl_comm_result}")
        
    def sync_write_torque(self, status,ids):
        data = self.divide_to_bytes(status, LEN_ENABLE_TORQUE)
        self.groupSyncWriteTorque.clearParam()
        for id in ids:
            result = self.groupSyncWriteTorque.addParam(id, data)
            if not result:
                self.get_logger().error(f"Failed to add parameter for id {id}")
                raise RuntimeError(f"Failed to add parameter for id {id}")
        dxl_comm_result = self.groupSyncWriteTorque.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to sync write torque: {dxl_comm_result}")
            raise RuntimeError(f"Failed to sync write torque: {dxl_comm_result}")
        
    def sync_write_pos(self, status,ids):
        data = self.divide_to_bytes(status, LEN_GOAL_POS)
        self.groupSyncWriteGoalPos.clearParam()
        for id in ids:
            result = self.groupSyncWriteGoalPos.addParam(id, data)
            if not result:
                self.get_logger().error(f"Failed to add parameter for id {id}")
                raise RuntimeError(f"Failed to add parameter for id {id}")
        dxl_comm_result = self.groupSyncWriteGoalPos.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to sync write pos: {dxl_comm_result}")
            raise RuntimeError(f"Failed to sync write pos: {dxl_comm_result}")
        
    def sync_write_cur(self, status,ids):
        data = self.divide_to_bytes(status, LEN_GOAL_CUR)
        self.groupSyncWriteGoalCur.clearParam()
        for id in ids:
            result = self.groupSyncWriteGoalCur.addParam(id, data)
            if not result:
                self.get_logger().error(f"Failed to add parameter for id {id}")
                raise RuntimeError(f"Failed to add parameter for id {id}")
        dxl_comm_result = self.groupSyncWriteGoalCur.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Failed to sync write cur: {dxl_comm_result}")
            raise RuntimeError(f"Failed to sync write cur: {dxl_comm_result}")
        
    def divide_to_bytes(self, value, length)->list[int]:
        if isinstance(value, list):
            raise NotImplementedError()
            return [self.divide_to_bytes(v, length) for v in value]
        if length == 1:
            return [value & 0xFF]
        elif length == 2:
            return [DXL_LOBYTE(value), DXL_HIBYTE(value)]
        elif length == 4:
            return [
                DXL_LOBYTE(DXL_LOWORD(value)),
                DXL_HIBYTE(DXL_LOWORD(value)),
                DXL_LOBYTE(DXL_HIWORD(value)),
                DXL_HIBYTE(DXL_HIWORD(value))
            ]
        else:
            self.get_logger().error(f"Invalid length {length}")
            return []
        
    @staticmethod
    def dxl_to_degree(positions):
        MAX_POSITION_VALUE = 4095
        CENTER_POSITION = 2047
        if isinstance(positions,np.ndarray):
            angle=np.zeros_like(positions)
            for i , position in enumerate(positions):
                angle[i] = ((position - CENTER_POSITION) / MAX_POSITION_VALUE) * 360
            return angle
        else:
            angle = ((positions - CENTER_POSITION) / MAX_POSITION_VALUE) * 360
            return angle
    
    @staticmethod
    def degree_to_dxl(positions):
        MAX_POSITION_VALUE = 4095
        CENTER_POSITION = 2047
        if isinstance(positions,np.ndarray):
            dxl=np.zeros_like(positions)
            for i , position in enumerate(positions):
                dxl[i] = (position/360.0*MAX_POSITION_VALUE)+CENTER_POSITION
            return dxl
        else:
            dxl=int((positions/360.0*MAX_POSITION_VALUE)+CENTER_POSITION)
            return dxl
        
    @staticmethod
    def dxl_to_radian(position):
        dxl_to_degree = DxlMasterArm.dxl_to_degree(position)
        return dxl_to_degree * DEG2RAD
    
    @staticmethod
    def radian_to_dxl(position):
        dxl = DxlMasterArm.degree_to_dxl(position/DEG2RAD)
        return dxl 
    
    def get_joint_positions(self) -> np.ndarray:
        """Return cached joint positions as [right*6, left*6] in radians.
        Updated at 25Hz by sync_read_position timer callback."""
        return np.concatenate([self.q['right_arm'], self.q['left_arm']])

    def shutdown(self):
        self.sync_write_led(0,self.all_ids())
        
    @staticmethod
    def uint16_to_int16(value):
        value = np.uint16(value)  # explicitly convert input to uint16
        return int(np.int16(value) )   # numpy automatically handles wrap-around
    
    @staticmethod
    def uint32_to_int32(value):
        value = np.uint32(value)  # explicitly convert input to uint32
        return int(np.int32(value) )   # numpy automatically handles wrap-around
  
import rclpy
    
def main(args=None):
    rclpy.init(args=args)
    node = DxlMasterArm(['right_arm','left_arm'])
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()