#키보드 입력 수신측에 전송 테스트트
import serial

def main():
    try:
        # UART 설정
        ser = serial.Serial(
            port='/dev/ttyAMA0',  # UART 포트
            baudrate=9600,       # Baud rate 설정
            bytesize=serial.EIGHTBITS,  # 데이터 비트: 8
            parity=serial.PARITY_NONE,  # 패리티: None
            stopbits=serial.STOPBITS_ONE,  # 스톱 비트: 1
            timeout=1           # 타임아웃 설정
        )

        if not ser.is_open:
            ser.open()

        print("UART 통신이 시작되었습니다. 'exit' 입력 시 종료됩니다.")
        while True:
            # 사용자 입력 받기
            user_input = input("전송할 문자열 입력: ")
            if user_input.lower() == 'exit':
                print("프로그램을 종료합니다.")
                break

            # 입력받은 문자열을 UART로 전송
            ser.write(user_input.encode('utf-8')+"\n")
            print(f"전송: {user_input}")

    except serial.SerialException as e:
        print(f"UART 초기화 오류: {e}")

    finally:
        if ser.is_open:
            ser.close()
        print("UART 포트를 닫았습니다.")

if __name__ == "__main__":
    main()
