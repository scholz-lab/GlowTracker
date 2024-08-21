---
title: 3. Build your own
layout: default
nav_order: 4
---

# Build your own GlowTracker

<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/macroscope_3d_parts.png" alt="All parts">
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/dual-color-photo.png" alt="After assembly">
          </figure>
        </td>
      </tr>
    </table>

## Steps for assembling your macroscope 
<p align="justify">Assuming you have all the parts from the <a href="https://scholz-lab.github.io/GlowTracker/List%20of%20parts/List_of_parts.html"><i>parts list</i></a> available, you should be able to assemble the macroscope in about two hours and have a fully functional tool.</p>

1. [Stage and base](#stage-base)
2. [Lightpath](#lightpath)
3. [Filters](#filters)
4. [Adjust field-of-view for dual color imaging ](#dualview-alignment) 
5. [Install the software](#install-software) 

### Tools
You should have a metric Thorlabs Balldriver & Hex Key Kits, and an SM1 spanner wrench handy. A tiny flat-head screwdriver is useful for adjusting the camera orientation.

## Stage and base <a name="stage-base"></a>
<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/0 - stage all parts.jpg" alt="Stage parts">
            <figcaption>Find these parts. They are required for stage assembly.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/finish.jpg" alt="After stage assembly">
            <figcaption>This is the result after successful stage assembly.</figcaption>
          </figure>
        </td>
      </tr>
    </table>
  
### Building the frame
<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/1.jpg" alt="Breadboard">
            <figcaption>1. Screw the four small posts into the bottom of the breadboard.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/2.jpg" alt="with feet">
            <figcaption>2. Fix the base plate to the breadboard using M6 screws.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/4 - connect to power.jpg" alt="stage connected to power">
            <figcaption>3. Connect the stage to power using the power cable. </figcaption>
          </figure>
        </td>
            <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/5 - move for space under.jpg" alt="stage moved aside">
            <figcaption>4. Use the knob to move the stage and expose the screw holes.</figcaption>
          </figure>
        </td>
      </tr>
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/3.jpg" alt="Stage attached to breadboard">
            <figcaption>5. Add the first stage axis by screwing it into the holder.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/6.jpg" alt="two stages attached to breadboard">
            <figcaption>6. Add the second stage axis by screwing it into the carriage of the first axis.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/7.jpg" alt="three stages attached to breadboard.">
            <figcaption>7. Add the final stage (vertical) by fixing the bottom end to the carriage of the second axis.</figcaption>
          </figure>
        </td> 
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/13 - insert merge plate.jpg" alt="adapter plate for optics attached">
            <figcaption>8.Fix the black adapter plate to the carriage of the vertical stage.</figcaption>
          </figure>
        </td> 
      </tr>  
    </table>

### Daisy-chaining the stage
<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/8.jpg" alt="Stage wiring" >
            <figcaption>1. Daisy chains the stages by connecting the port labeled 'next' of the first axis to the 'prev' port of the second axis.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/10.jpg" alt="Stage wiring" >
            <figcaption>2. Connect the 'next' port of axis 2 to 'prev' of axis 3. All stages should light up green when the first is connected to the power source.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/11 - connect computer usb line.jpg" alt="USB connection" >
            <figcaption>3. Connect the first axis to the USB cable by plugging it into the port labeled 'prev'.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/12 - connect power.jpg" alt="After stage assembly">
            <figcaption>4.  All stages should light up green when the first is connected to the power source.</figcaption>
          </figure>
        </td>
      </tr>
    </table>

## Lightpath - hardware only <a name="lightpath"></a>

### Hardware 
<p align="justify">
  In this section, you will add the lightpath to the stages. We will first assemble the structure and add the filters and dichroics in the next step.
</p>

<p align="justify">
  <i><b>Note:</b> If you order the lenses we suggest, they come pre-mounted. However, in case you are using lenses you already have you need to mount them in a short (10 mm) lens tube using a retaining ring. Refer to the lightpath diagram and the photos for correct lens orientation.</i>
</p>

<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
             <img src="custom_assets/images/2 - optics/all_parts_labeled.jpg" alt="After lightpath assembly" >
            <figcaption>Find these parts. You can lay out the filters on optical paper. Dichroics are <b>not</b> labeled - so be sure not to switch them!</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/3 - Finish/complete - side.jpg" alt="After lightpath assembly" >
            <figcaption>When you finish this section you should have the macroscope without filters fully assembled.</figcaption>
          </figure>
        </td>
      </tr>
    </table>


### LED assembly
In this section, we assemble the white light LED. The optics are used to collimate the light and focus a slit onto the focal plane of the sample, restricting illumination to a small area. This will allow us to later separate the two colors onto the camera chip.
  <table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/1.jpg" alt="white LED" >
          <figcaption>1. Start with the white LED.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/2.jpg" alt="lens tube added" >
          <figcaption>2. Add a lens tube for spacing.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/3.jpg" alt="short focal length lens">
          <figcaption>3. Add the 16 mm focal length lens. </figcaption>
        </figure>
      </td>
          <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/4 - secure with ring.jpg" alt="retaining ring" >
          <figcaption>4. Secure the lens with a retaining ring. </figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/5 - put in connector.jpg" alt="Stage attached to breadboard" >
          <figcaption>5. Add another lens tube as a spacer.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/6.jpg" alt="two stages attached to breadboard">
          <figcaption>6. Identify the slit and the adjustable element (thicker black tube).</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/7.jpg" alt="adjustable element.">
          <figcaption>7. Secure the slit inside the adjustable element using a retaining ring.</figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/8.jpg" alt="adapter plate for optics attached" >
          <figcaption>8. Screw the adjustable element into the lens tube from step 5.</figcaption>
        </figure>
      </td> 
    </tr>
     <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/9.jpg" alt="spacer" >
          <figcaption>9. Add a short lens tube as a spacer.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/10.jpg" alt="50 mm lens" >
          <figcaption>10. Add the 50mm focal length lens. </figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/14 - complete.jpg" alt="adjustable element." >
          <figcaption>11. The final assemble should look like this.</figcaption>
        </figure>
      </td> 
    </tr>  
  </table>

### Image-splitter assembly
<p align="justify">
  The image splitter is the heart of the macroscope. Mechanics matter as a crooked assembly will lead to issues with the image quality. Before starting, lay out all the parts required. Pay attention to the cube orientations and follow the pictures exactly. 
</p>


<table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/1 - connector on bottom of cube 2.jpg" alt="adjustable element." >
          <figcaption>1. Identify one of the 3 DFM1 cubes and attach the cage cube adapter.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/2.jpg" alt="lens tube added" >
          <figcaption>2. Cube after adding connector. Note <b>orientation</b>!</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/3 - connect cube 2 ontop cube 1.jpg" alt="short focal length lens" >
          <figcaption>3. Connect another DFM1 cube by using the black screws from the connector pack. </figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/4.jpg" alt="retaining ring" >
          <figcaption>4. Close the left holes with two SM1CP2 caps and screw in 4 ER025 cage assembly rods into the top cube. </figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/5.jpg" alt="Stage attached to breadboard" >
          <figcaption>5. Connect the triangular mirror using the set screws. Remove the tiny screws from the ends of the ER025 rods so they fit fully into the holes of the mirror.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/6.jpg" alt="two stages attached to breadboard">
          <figcaption>6. Screw 4 ER025 rods into the right side of the middle cube, again removing the set screws on the other side.</figcaption>
        </figure>
      </td>
       </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/7 - insert_cage_connectors.jpg" alt="adjustable element." >
          <figcaption>7. Grab the third DFM1 cube and screw 4 ER025 rods into the bottom. </figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/8.jpg" alt="adapter plate for optics attached" >
          <figcaption>8. Secure to the triangular mirror using set screws. Ensure the mirror is securely attached and not crooked.</figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/9.jpg" alt="spacer" >
          <figcaption>9. Add 4 ER025 rods to the cube-mirror from 8. such that the two halves of the image splitter look as shown.</figcaption>
        </figure>
      </td>
    </tr>  
     <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/10.jpg" alt="spacer" >
          <figcaption>10. Loosen the set screws in the mirrors and slide the halves together. Tighten the set screws to secure the halves. If all parts are assembled straight, this should not require force!</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/11.jpg" alt="50 mm lens" width="100%">
          <figcaption>11. Add 4 ER025 rods to the closed-off side of the image splitter for mounting to the stage.<br>Insert mirrors into mirror mounts.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/cam_1.jpg" alt="50 mm lens" >
          <figcaption>12. Connect the Basler camera to the Yongnuo objective using the Kipon adapter. </figcaption>
        </figure>
      </td>
    </tr>  
  </table>

### Finishing up
<table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/cam_2.jpg" alt="white LED" >
          <figcaption>1. Connect the camera-lens ensemble to the image splitter using the SM2A53 and SM1A2 - Adapter.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/finish.jpg" alt="After stage assembly" >
            <figcaption>2. Add the ERSCB connectors to the stage plate.</figcaption>
          </figure>
      </td>
      <td>
         <figure class="center-figure">
            <img src="custom_assets/images/3 - Finish/complete - side.jpg" alt="After lightpath assembly">
            <figcaption>3. Slide the image-splitter and camera assembly into the ERCSB connectors. Secure with the set screws.</figcaption>
          </figure>
      </td>
    </tr>
  </table>

## Lightpath - Filters and Dichroics <a name="filters"></a>
<p align="justify">
  In this section, you will handle a lot of expensive filters and dichroic mirrors. It is helpful to have optical cleaning supplies handy in case you touch a lens or filter. Always touch optics at the edge, as smudges are hard to remove and will degrade your image quality. Filters and dichroics have a preferred direction, so please pay attention to the orientation you put them in. If your image quality is not ideal, and you see reflections or double images, that may be due to a flipped filter.
</p>

The schematic of the filters is shown here:
<figure class="center-figure">
  <img src="custom_assets/images/light_path_2.png" alt="cube" width="75%">
  <figcaption>Refer to the schematic to identify which filters are in which of the cubes.</figcaption>
</figure>

### Filter cube 1 - Excitation light
<p align="justify">
  All filters will require similar steps. We will show the full process for the first cube. Repeat for Cubes 2 and 3. A video for instructions can be found <a href="https://youtu.be/qWIfiwuL-gQ"><i>here</i></a>.
</p>

For this cube, you will need one 488/561 nm dichroic and a dualband excitation filter 59011x.
<table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/10.jpg" alt="cube" >
          <figcaption>1. Take out the insert of the bottom cube.</figcaption>
        </figure>
      </td>
       <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/filtercubes/filtercube 1 - excitation/4.jpg" alt="cube" >
          <figcaption>2. Open the mirror insert using the two screws.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
            <img src="custom_assets/images/2 - optics/filtercubes/filtercube 1 - excitation/2.jpg" alt="cube" >
            <figcaption>3. Carefully insert the dichroic mirror 488/561 nm. The reflective surface should point towards the LED when the insert is slotted back into the cube </figcaption>
          </figure>
      </td>
    </tr>
    <tr>
       <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/filtercubes/filtercube 1 - excitation/3.jpg" alt="cube" >
          <figcaption>4. Close the insert back up.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
            <img src="custom_assets/images/2 - optics/filtercubes/filtercube 1 - excitation/1.png" alt="cube" >
            <figcaption>5. Add the excitation filter to the cube insert with the arrow pointing inward. Secure it with a retaining ring. The filter should sit in front of the LED when the insert is back in the light splitter. </figcaption>
        </figure>
      </td>
    </tr>
  </table>


### Filter cube 2 - Image splitter cube with emission filters
<p align="justify">
  Here you will need one of the two 561nm longpass dichroics, a 618/50 nm bandpass for the red image and a 520/36 bandpass for the green image.
</p>

<p align="justify">
  Follow the steps above to secure the dichroic. Add the two filters, the red at the top and the green at the right side of the filter. The 'top' and 'right' orientation refers to the cube's placement when you are looking at it in the macroscope. Since the filters are from Semrock and Edmund Optics, the direction of the arrows should follow the direction of the light, i.e. outward of the cube and toward the camera as depicted below.
</p>
<table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/filtercubes/filtercube 2 - split/2.jpg" alt="618/50" >
          <figcaption>Filter 618/50 nm direction</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/filtercubes/filtercube 2 - split/3.jpg" alt="520/36" >
          <figcaption>Filter 520/36 nm direction</figcaption>
        </figure>
      </td>
    </tr>
</table>


### Filter cube 3 - Image splitter cube without emission filters
<p align="justify">
  This cube combines the red and green images. You will only need the final  561nm longpass dichroic. Secure it in the cube as described above. Make sure the reflective side is oriented correctly.
</p>

## Adjust field-of-view for dual color imaging <a name="dualview-alignment"></a>
<p align="justify">
To adjust the field-of-view for dual color imaging we first need to align the mirrors such that dual color images are projected onto the camera side by side without overlapping. Before starting, ensure that your GlowTracker settings are ‘dual color’ and the view is set to ‘splitted’. Turn off all environment lights to avoid seeing double images. Follow the instructions below to align both images.
You will need dual colored samples. We found that fluorescent tape or even fluorescent markers on a glas slide work well.

<ol>
  <li>
  Positioning of first color channel (here red):
    <ol type="a">
    <li>Concentrate on the red channel first, for this block the green channel by removing the image splitter cube (the middle cube)</li>
    <li>Focus on a sample using the z-axis of the stage. A white paper with small printed letters or a barcode works well</li>
    <li>Focus the slit, by rotating the adjustable element, until its edges come sharp in focus (the sample needs to be in focus too)</li>
    <li>Loosen the screws of the adapter that holds the camera in place, rotate the camera so that the slit is orientated vertical to the image of the camera</li>
    <li>Use screws at the mirror mount of the red channel to move the image, so that the highest intensity is in the center of one of the sides. Note that there are 3 ways to adjust the mirror: 2 screws and one hole requiring a hex key that moves the image diagonally.</li>
    <li>Put back the middle cube</li>
    </ol>
  </li>
  <li>
  Alignment of second color channel (here green):
    <ol type="a"> 
    <li>Block the light of the red channel if possible, e.g. by putting a piece of paper in the red light path.</li>
    <li>Use the screws of the mirror mount of the green channel to first find the edge of the slit, and second position the green image as in 1. e</li>
    <li>Unblock the red light</li>
    <li>Adjust the position of the green image so that the structures are approx. at the same position (in the split view) as in the red channel</li>
    </ol>
  </li>
  <li>
  Test the alignment by using fluorescent samples (e.g. fluorescent tape)
    <ol type="a">
    <li>Test if the split view shows images only in the expected wavelength i.e., red samples show up on one side and green samples on the other side.</li>
    <li>Check alignment on sample with both wavelength (e.g. use yellow fluorescent tape, or overlay red and green fluorescent tape)</li>
    <li>Adjust alignment if necessary, you can also use the merged view (-> GUI/ Settings)</li>
    <li>Try to keep the highest intensity in the center of the images</li>
    </ol>
  </li>
  <li>
  Dual color calibration
    <ol type="a">
    <li>Open the calibration menu at the right of the GlowTracker GUI</li>
    <li>Open the dual color calibration mode and press calibrate</li>
    <li>If the calibration is not good try to find either a better part of the sample or try to focus both sides, repeat the calibration</li>
    </ol>
  </li>
  <li>
  Check alignment on biological samples
  </li>
</ol>


<figure class="center-figure">
  <img src="custom_assets/images/4 - dual color hardware adjustment/slitalignment_comp.jpg" alt="slit alignment" width="100%">
  <figcaption>Example process of slit alignment. Numbering refers to steps in instruction.</figcaption>
</figure>
Note: You can also manually change the calibration parameters in the settings, this can be helpful if you have problems with automatic calibration.
</p>

## Install the software <a name="install-software"></a>
<p align="justify">
  The Macroscope GUI is designed for manual and automated tracking, offering a Kivy app interface for two USB devices. Users are encouraged to adjust camera parameters using BASLER's pylon software to avoid duplicating configuration efforts. GUI functions are primarily in the Kivy file, while device functionality is delegated to specific modules. 

  To work with the GUI, you will need to do the following:
</p>
1. Create an environment
2. Install BASLER package
3. Install Zaber Launcher <i>(optional)</i>
4. Start the GUI by <code>python GlowTracker.py</code>

<p align="justify">
  You can find more detailed instructions in the <a><i>Software installation</i></a> section.
</p>
