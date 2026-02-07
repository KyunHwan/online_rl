#!/usr/bin/env python3

import sys
import time
import threading
from pynput import keyboard
from pynput.keyboard import Key, Listener
from rclpy.node import Node
from std_msgs.msg import String
import rclpy

class IOHandler(Node):
    def __init__(self):
        super().__init__('io_handler')
        self.pressed_keys = set()
        self.listener = None
        self.is_running = False
        
        self.declare_parameter('robot_id', 'cashy')
        self.robot_id = self.get_parameter('robot_id').value
        
        self.publisher_ = self.create_publisher(String, f'/igris_b/{self.robot_id}/io_event', 10)
        
        # Pre-create message object to avoid repeated allocation
        self.msg = String()
        
    def on_press(self, key):
        try:
            # Only process if not already pressed (avoid duplicate events)
            if key in (Key.shift, Key.shift_l, Key.shift_r):
                pass
            else:
                if key not in self.pressed_keys:
                    self.pressed_keys.add(key)
                    key_str = str(key)
                    self.msg.data = key_str
                    self.publisher_.publish(self.msg)
                
                # Handle ESC key
                if key == Key.esc:
                    print("ESC key pressed - stopping listener")
                    self.stop_listening()
                    return False
                    
        except AttributeError:
            # Handle special keys more efficiently
            key_str = str(key)
            self.msg.data = key_str
            self.publisher_.publish(self.msg)
            
    def on_release(self, key):
        # Remove from pressed keys set for better memory management
        
        self.pressed_keys.clear()
            
    def start_listening(self):
        print("Starting optimized keyboard listener...")
        print("Press ESC to stop listening")
        
        self.is_running = True
        
        # Use daemon thread for better cleanup
        self.listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release,
            suppress=False
        )
        
        self.listener.start()
        
    def stop_listening(self):
        if self.listener:
            self.listener.stop()
            self.is_running = False
            self.pressed_keys.clear()  # Clean up memory
            print("Keyboard listener stopped")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        handler = IOHandler()
        handler.start_listening()
        
        print("IO Handler node started. Press Ctrl+C to exit.")
        rclpy.spin(handler)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'handler' in locals():
            handler.stop_listening()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
        
