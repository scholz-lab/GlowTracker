---
title: Performance benchmark
layout: default
---
# Performance benchmark

<!-- https://wiki.geogebra.org/en/Reference:GeoGebra_Apps_Embedding -->
<!-- https://www.geogebra.org/m/umrknut3 -->
<!-- https://www.geogebra.org/m/sehv2qc9 -->
<!-- https://geogebra.github.io/integration/ -->

<div>
    <meta name=viewport content="width=device-width,initial-scale=1">  
    <meta charset="utf-8"/>
    <script src="https://www.geogebra.org/apps/deployggb.js"></script>
    <div id="ggb-element" style="height: 500px; width: auto;"></div>
    <script type="text/javascript">

        var containerRect = document.getElementById('ggb-element').getBoundingClientRect();
        
        var params = {
            appName: "geometry", 
            material_id:"eb7mqevw",
            allowUpscale: true,
            autoHeight: true,
            width: containerRect.width,
            height: containerRect.height,
            showToolBar: false, 
            showMenuBar: false,
            showAlgebraInput: false, 
            showToolBarHelp: false,
            showResetIcon: true,
            errorDialogsActive: true,
            useBrowserForJS: false,
        };

        var ggbApplet = new GGBApplet(params, true);
        
        window.addEventListener("load", function() { 
            ggbApplet.inject('ggb-element');
        });
    </script>
</div>