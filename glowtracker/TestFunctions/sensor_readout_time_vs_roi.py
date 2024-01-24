from pypylon import pylon, genicam
import numpy as np
import math

if __name__ == '__main__':

    # Init camera
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice())

    camera.Open()

    # Variables
    FPS = 30
    exposureTime = 0.03
    areaStepSize = 100000

    # Set the variables into camera
    #   FPS
    camera.AcquisitionFrameRateEnable = True
    camera.AcquisitionFrameRate = float(FPS)
    #   Exposure
    camera.ExposureTime.SetValue( exposureTime * 1e6 )

    #   Image ROI
    # Set offsets to 0
    camera.OffsetX.SetValue(0)
    camera.OffsetY.SetValue(0)
    
    # Gather min, max width and height
    maxWidth = camera.WidthMax.GetValue()
    maxHeight = camera.HeightMax.GetValue()
    minWidth = 376  # Camera spec acA3088-57um: 376, 320
    minHeight = 320
    aspectRatio = minWidth / minHeight

    if minWidth / minHeight != maxWidth / maxHeight:
        print(f'Min aspect and Max aspect is not the same.')
        print(f'Min aspect: {minWidth / minHeight}, Max aspect: {maxWidth/maxHeight}')

    currWidth = minWidth
    currHeight = minHeight
    currArea = minWidth * minHeight
    maxArea = maxWidth * maxHeight

    # Data record holder
    custom_dtype = {
        'names' : ('width', 'height', 'area', 'sensor_readout_time', 'resulting_spf'),
        'formats' : ('i', 'i', 'i', 'd', 'd')
    }
    data = np.zeros(1, custom_dtype)

    lastIterFlag = False
    
    # Iteratively increase the readout area 
    while currArea <= maxArea or lastIterFlag:

        # Set Camera ROI
        camera.Width.SetValue( currWidth )
        camera.Height.SetValue( currHeight )

        # Record sensor readout time and resulting SPF
        newData = np.zeros(1, custom_dtype)
        newData['width'][0] = currWidth
        newData['height'][0] = currHeight
        newData['area'][0] = currArea
        newData['sensor_readout_time'][0] = camera.SensorReadoutTime.GetValue()
        newData['resulting_spf'][0] = 1.0/camera.ResultingFrameRate.GetValue()
        data = np.vstack([data, newData])

        # Increase the area
        currArea += areaStepSize
        # currArea *= 2

        # Compute width and height from a given area and aspect ratio
        currWidth = round( math.sqrt( currArea * aspectRatio ) )
        currHeight = round( currWidth / aspectRatio )

        # The difference between current value and the minimum value must be 
        #   divisible by 4 for widht, and 2 for height
        #   (camera's requirement)
        diffWidthModulo = ( currWidth - minWidth ) % 4
        if diffWidthModulo != 0:
            currWidth += 4 - diffWidthModulo

        diffHeightModulo = ( currHeight - minHeight ) % 2
        if diffHeightModulo != 0:
            currHeight += 2 - diffHeightModulo
        
        # Cap maximum
        currWidth = min( maxWidth, currWidth )
        currHeight = min( maxHeight, currHeight )

        # Update area 
        currArea = currWidth * currHeight

        if not lastIterFlag:
            # If the next iter exceed maximum ROI then cap to maximum ROI
            # and set as last iteration
            if currArea >= maxArea or currWidth >= maxWidth or currHeight >= maxHeight :
                currArea = maxArea
                currWidth = maxWidth
                currHeight = maxHeight
                lastIterFlag = True
        else:
            break
    
    # Delete first zero row
    data = np.delete(data, 0, 0)

    # Print Column header
    column_name_str = ','.join( data.dtype.names )
    print(column_name_str)

    # Print data in each row
    for data_row in data:
        row_items = data_row.item()
        row_items_str = ','.join( str(i) for i in row_items )
        print(row_items_str)
    