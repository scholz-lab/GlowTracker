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


