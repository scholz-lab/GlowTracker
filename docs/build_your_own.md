---
title: 3. Build your own
layout: default
nav_order: 3
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

## Steps for assembling your microscope
Assuming you have all the parts from the parts list [TODO: link] handy, the assembly should take about 2 h to go from parts to fully functional microscope. 

1. [Stage and base](#stage-base)
2. [Lightpath](#lightpath)
3. [Filters](#filters)
4. [Adjust field-of-view for dual color imaging ](#dual-color-fov) 
5. [Dual-color calibration (optional).](#dual-color-calibration) 
6. [Install the software](#install-software) 

### Tools
You should have a metric Thorlabs Balldriver & Hex Key Kits, and an SM1 spanner wrench handy. A tiny flat head screwriver is useful for adjusting the camera orientation.

## Stage and base <a name="stage-base"></a>
<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/0 - stage all parts.jpg" alt="Stage parts" width=100%>
            <figcaption>Find these parts. They are required for stage assembly.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/finish.jpg" alt="After stage assembly" width =73%>
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
            <img src="custom_assets/images/1 - stage/1.jpg" alt="Breadboard" width=100%>
            <figcaption>1. Screw the four small posts into the bottom of the breadboard.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/2.jpg" alt="with feet" width=90%>
            <figcaption>2. Fix the base plate to the breadboard using M6 screws.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/4 - connect to power.jpg" alt="stage connected to power" width=100%>
            <figcaption>3. Connect the stage to power using the power cable. </figcaption>
          </figure>
        </td>
            <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/5 - move for space under.jpg" alt="stage moved aside" width=100%>
            <figcaption>4. Use the knob to move the stage and expose the screw holes.</figcaption>
          </figure>
        </td>
      </tr>
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/3.jpg" alt="Stage attached to breadboard" width=100%>
            <figcaption>5. Add first stage axis by screwing into holder.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/6.jpg" alt="two stages attached to breadboard" width=100%>
            <figcaption>6. Add the second stage axis by screwing it into the carriage of the first axis.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/7.jpg" alt="three stages attached to breadboard." width=100%>
            <figcaption>7. Add the final stage (vertical) by fixing the bottom end to the carriage of the second axis.</figcaption>
          </figure>
        </td> 
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/13 - insert merge plate.jpg" alt="adapter plate for optics attached" width=63%>
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
          <img src="custom_assets/images/1 - stage/8.jpg" alt="Stage wiring" width=100%>
          <figcaption>1. Daisy chain the stages by connecting the port labelled 'next' of the first axis to the 'prev' port of the second axis.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/1 - stage/10.jpg" alt="Stage wiring" width=100%>
          <figcaption>2. Connect the 'next' port of axis 2 to 'prev' of axis 3. All stages should light up green when the first is connected to the power source.</figcaption>
        </figure>
      </td>
       <td>
        <figure class="center-figure">
          <img src="custom_assets/images/1 - stage/11 - connect computer usb line.jpg" alt="USB connection" >
          <figcaption>3. Connect the first axis to the USB cable by plugging it into the port labelled 'prev'.</figcaption>
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
    In this section you will add the lightpath to the stages. We will first assemble the structure and add the filters and dichroics in the next step.

      Note: If you order the lenses we suggest, they come pre-mounted. However, in case you are using lenses you already have you need to mount them in a short (10 mm) lens tube using a retaining ring. Refer to the lightpath diagram and the photos for correct lens orientation.

<table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
             <img src="custom_assets/images/2 - optics/all parts.jpg" alt="After lightpath assembly" width =200%>
            <figcaption>Find these parts. You can lay out the filters on optical paper. Dichroics are *not* labelled - so be sure not to switch them!.</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/3 - Finish/complete - side.jpg" alt="After lightpath assembly" width =70%>
            <figcaption>When you finish this section you should have the macroscope without filters fully assembled.</figcaption>
          </figure>
        </td>
      </tr>
    </table>


### LED assembly
In this section we assemble the white light LED. The optics are use to collimate the light and focus a slit onto the focal plane of the sample, restricting illumination to a small area. This will allow us to later separate the two colors onto the camera chip.
  <table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/1.jpg" alt="white LED" width=100%>
          <figcaption>1. Start with the white LED.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/2.jpg" alt="lens tube added" width=100%>
          <figcaption>2. Add a lens tube for spacing.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/3.jpg" alt="short focal length lens" widh=100%>
          <figcaption>3. Add the 16 mm focal length lens. </figcaption>
        </figure>
      </td>
          <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/4 - secure with ring.jpg" alt="retaining ring" width=100%>
          <figcaption>4. Secure the lens with a retaining ring. </figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/5 - put in connector.jpg" alt="Stage attached to breadboard" width=100%>
          <figcaption>5. Add another lens tube as spacer.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/6.jpg" alt="two stages attached to breadboard" width=100%>
          <figcaption>6. Identify the slit and the adjustable element (thicker black tube).</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/7.jpg" alt="adjustable element." width=100%>
          <figcaption>7. Secure the slit inside the adjustable element using aretaining ring..</figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/8.jpg" alt="adapter plate for optics attached" width=63%>
          <figcaption>8. screw the adjustable element into the lens tube from step 5.</figcaption>
        </figure>
      </td> 
    </tr>  
     <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/9.jpg" alt="spacer" width=100%>
          <figcaption>9. Add a short lens tube as spacer.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/10.jpg" alt="50 mm lens" width=100%>
          <figcaption>10. Add the 50mm focal length lens. </figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2 - optics/14 - complete.jpg" alt="adjustable element." width=100%>
          <figcaption>11. The final assemble should look like this.</figcaption>
        </figure>
      </td> 
    </tr>  
  </table>

### Image-splitter assembly
    The image splitter is the heart of the microscope. Mechanics matter, as a crooked assembly will lead to issues with the image quality. Before starting, lay out all parts required. Pay attention to the cube orientations and follow the pictures exactly. 


<table class="equal-column-table">
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/cam_2.jpg" alt="adjustable element." width=100%>
          <figcaption>1. Connect the Kipon adapter to the objective and screw the camera in. Add the SM2A53 - Adapter and SM1A2 - Adapter to the other side of the objective to connect to the image splitter.
</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/2.jpg" alt="lens tube added" width=100%>
          <figcaption>2. Cube after adding connector. Note orientation!</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/3 - connect cube 2 ontop cube 1.jpg" alt="short focal length lens" widh=100%>
          <figcaption>3. Connect another DFM1 cube by using the black screws from the connector pack. </figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/4.jpg" alt="retaining ring" width=100%>
          <figcaption>4. Close the left holes with two SM1CP2 caps and screw in 4 ER025 cage assembly rods into the top cube. </figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/5.jpg" alt="Stage attached to breadboard" width=100%>
          <figcaption>5. Connect the triangular mirror using the set screws. Remove the tiny screws from the ends of the ER025 rods so they fit fully into the holes of the mirror.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/6.jpg" alt="two stages attached to breadboard" width=100%>
          <figcaption>6. Screw 4 ER05 rods into the right side of the middle cube, again removing the set screws on the other side.</figcaption>
        </figure>
      </td>
       </tr>
    <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/7 - insert_cage_connectors.jpg" alt="adjustable element." width=100%>
          <figcaption>7. Grab the third DFM1 cube and screw 4 ER025 rods into the bottom. </figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/8.jpg" alt="adapter plate for optics attached" width=63%>
          <figcaption>8. Secure to the triangular mirror using set screws. Ensure the mirror is securly attached and not crooked.</figcaption>
        </figure>
      </td> 
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/9.jpg" alt="spacer" width=100%>
          <figcaption>9. Add 4 ER05 rods to the cube-mirror from 8. such that the two halves of the image splitter look as shown.</figcaption>
        </figure>
      </td>
    </tr>  
     <tr>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/10.jpg" alt="spacer" width=100%>
          <figcaption>10. Loosen the set screws in the mirrors and slide the halves together. Tighten the set screws to secure the halves. If all parts are assembled straight, this should not require force!</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/11.jpg" alt="50 mm lens" width=100%>
          <figcaption>11. Add 4 ER rods to the closed-off side of the image splitter for mounting to the stage. </figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
          <img src="custom_assets/images/2a-lightpath/cam_1.jpg" alt="50 mm lens" width=100%>
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
          <img src="custom_assets/images/2a-lightpath/cam_2.jpg" alt="white LED" width=100%>
          <figcaption>1. Connect the camera-lens ensemble to the image splitter using the SM2A53 and SM1A2 - Adapter.
.</figcaption>
        </figure>
      </td>
      <td>
        <figure class="center-figure">
            <img src="custom_assets/images/1 - stage/finish.jpg" alt="After stage assembly" width =73%>
            <figcaption>2. Add the ERSCB connectors to the stage plate.</figcaption>
          </figure>
      </td>
      <td>
         <figure class="center-figure">
            <img src="custom_assets/images/3 - Finish/complete - side.jpg" alt="After lightpath assembly" width =70%>
            <figcaption>3. Slide the image-splitter and camera assembly into the ERCSB connectors. Secure with the set screws.</figcaption>
          </figure>
        </td>
      </td>
    </tr>
  </table>

## Lightpath - Filters and Dichroics <a name="filters"></a>
    Note: In this section you will handle a lot of expensive filters and dichroic mirrors. It is helpful to have optical cleaning supplies handy in case you touch a lens or filter. Always touch optics at the edge, as smudges are hard to remove and will degrade your image quality. Filters and dichroics have a preferred direction, so please pay attention to the orientation you put them in. If you image quality is not ideal, and you see reflections or double-images, that may be due to a flipped filter.

### Filter cube 1 - Excitation light

### Filter cube 2 - Image splitter cube with emission filters
### Filter cube 3 - Image splitter cube without emission filters

## Adjust field-of-view for dual color imaging <a name="dualview-alignment"></a>

## Install the software <a name="install-software"></a>
