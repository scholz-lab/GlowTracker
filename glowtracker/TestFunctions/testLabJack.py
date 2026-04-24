import u3
import time
if __name__ == '__main__':

    # Connect to device
    device = u3.U3()
    # print(device.configU3())
    device.debug = True


    # Get calibration data
    cabDat = device.getCalibrationData()
    # print(cabDat)

    # Send analog output
    #   Voltage 1.5, channel DAC0, 8 Bits
    dac0Val = device.voltageToDACBits(volts= 1, dacNumber= 0, is16Bits= False)
    print("dac0Val", dac0Val)

    dac1Val = device.voltageToDACBits(volts= 2, dacNumber= 1, is16Bits= False)
    print("dac1Val", dac1Val)

    #   Create a Feedback command 
    dac0Command = u3.DAC0_8(dac0Val)
    dac1Command = u3.DAC1_8(dac1Val)
    print("dac0Command", dac0Command)
    print("dac1Command", dac1Command)

    #   Send and get feedback
    feedback = device.getFeedback(dac0Command, dac1Command)
    print("feedback", feedback)

    # Get analog input
    # takes about 30ms for the output to reach target V
    time.sleep(0.030)
    # Config FIO0 to be analog
    device.configAnalog(u3.FIO0)
    ain0Command = u3.AIN(0, 31, True)

    feedback = device.getFeedback(ain0Command)
    print("AIN0 feedback", feedback)

    #   Convert analog input bits to voltage
    ainValue = device.binaryToCalibratedAnalogVoltage(bits= feedback[0], isLowVoltage= True, isSingleEnded= True, isSpecialSetting= False, channelNumber= 0)
    print("AIN0 vol", ainValue)

    #   Can also read using this helper func
    newAinValue = device.getAIN(posChannel=0, negChannel=31, longSettle=False, quickSample=False)
    print("Another AIN0 vol", newAinValue)

    #
    # Output and Input digital signals
    # 
    # Set FIO to digital 
    device.configDigital(u3.FIO0, u3.FIO2)
    # Set digital output at FI00
    device.setFIOState(fioNum= 0, state= 1)
    # device.setDOState(ioNum= 0, state= 1)

    # Get digital input at FI01
    inputState = device.getDIState(ioNum= 2)
    print(f"FIO2 Input state {inputState}")
    

    # Set digital output at FI00
    device.setDOState(ioNum= 0, state= 0)

    # Get digital input at FI01
    inputState = device.getDIState(ioNum= 2)
    print(f"FIO2 Input state {inputState}")


    


    