import serial

def read_uart():

    ser = serial.Serial(
        port='/dev/ttyAMA2', 
        baudrate=9600,   
        timeout=1  
    )

    try:
        print("Listening on UART (AMA2)...")
        while True:
            if ser.in_waiting > 0:  # ????? ??????? ??? ???
                data = ser.readline().decode('utf-8').strip()  # ?????? ?��? ?????
                print(f"Received: {data}")
                ser.write("0\r".encode('utf-8'))
                break
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()  # ???? ?? ??? ???

if __name__ == "__main__":
    while 1:
        read_uart()
