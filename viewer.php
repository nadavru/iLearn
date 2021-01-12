<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>iLearnML Graph Viewer</title>
  <script src="mathbox-bundle.js"></script>
  <script src="dat.gui.js"></script>

<!-- http://silentmatt.com/javascript-expression-evaluator/ -->
<script src="parser.js"></script>

  <link rel="stylesheet" href="mathbox.css">
  <meta name="viewport" content="initial-scale=1, maximum-scale=1">
	<style>
		#paramForm {

		}
	</style>
	<link rel = "icon" href ="title_icon.png" type = "image/x-icon">
</head>
<body>
<div id="paramForm">
	<table>
		<form>
				<tr>
					<td>
						Function:
						<input type="text" id="func" name="func" value="sin(x^2+y^2)">
					</td>
					<td>
						Style:
						<select id="style" >
							<option value="rb">red blue</option>
							<option value="rainbow">rainbow</option>
						  <option value="grayscale">grayscale</option>
						  <option value="solid blue">solid blue</option>
						</select>
					</td>
					<td>
						View wireframe:
						<select id="wireframe" >
						  <option value="true">yes</option>
						  <option value="false">no</option>
						</select>
					</td>
					<td>
						View shadow:
						<select id="shadow" >
							<option value="false">no</option>
							<option value="true">yes</option>
						</select>
					</td>
					<td>

					</td>
				</tr>
				<tr>
					<td colspan="4">
						xMax : <input type="text" id="xmax" value="3"> xMin : <input type="text" id="xmin" value="-3"> yMax : <input type="text" id="ymax" value="3"> YMin : <input type="text" id="ymin" value="-3"> zMax : <input type="text" id="zmax" value="0"> zMin : <input type="text" id="zmin" value="0">
					</td>
				</tr>
				<tr>
					<td colspan="4">
						startX : <input type="text" id="startx" value="0"> startY : <input type="text" id="starty" value="0"> startZ : <input type="text" id="startz" value="0">endX : <input type="text" id="endx" value="0"> endY : <input type="text" id="endy" value="0"> endZ : <input type="text" id="endz" value="0">
						<button type="button" onclick="updateGraph()">update Graph</button>
					</td>
				</tr>
		</form>
	</table>
</div>

		<table>
			<tr>
				<td>
					<button type="button" onclick="NextVec()">Next Step</button>
					<button type="button" onclick="PrevVec()">Previous Step</button>
				</td>

			</tr>
		</table>

  <script>
    document.getElementById("paramForm").style.display = "none";
    var mathbox = mathBox({
      plugins: ['core', 'controls', 'cursor', 'mathbox'],
      controls: {klass: THREE.OrbitControls}
    });
    if (mathbox.fallback) throw "WebGL not supported"

    var three = mathbox.three;
    three.renderer.setClearColor(new THREE.Color(0xFFFFFF), 1.0);
	var displayTrace = true;
	var view;

	var functionText = "sin(x^2+y^2)";

	var pointText = "(1,1)";
	var currVecNum=0;
	var points=[];
	var a = 1, b = 1;
	var	xMin = -3, xMax = 3, yMin = -3,	yMax = 3, zMin = -3, zMax = 3;
	var zAutofit = true;

	// start of updateGraph function ==============================================================
	var updateGraphFunc = function() {
		currVecNum=0;
		points=[];
		var hashParams = window.location.hash;
		if(hashParams.length != 0) {
			hashParams = hashParams.substr(1).split('&');// substr(1) to remove the `#`
			for (var i = 0; i < hashParams.length; i++) {
				var p = hashParams[i].split('=');
				if (p[0]=="points"){
					points=p[1].split('|');
				}else {
					if(p[0]=="hide"){
						var x = document.getElementById("paramForm");
						if(p[1]=="true"){
							x.style.display = "block";
						}else{
							x.style.display = "none";
						}
					}else {
						document.getElementById(p[0]).value = decodeURIComponent(p[1]);
					}
				}
			}
		}
		var zFunc = Parser.parse(document.getElementById("func").value.toLowerCase()).toJSFunction(['x', 'y']);
		var graphColorStyle = document.getElementById("style").value.toLowerCase();
		var wireframe = document.getElementById("wireframe").value.toLowerCase();
		var shadow = document.getElementById("shadow").value.toLowerCase();
		var xMax = parseFloat(document.getElementById("xmax").value);
		var yMax = parseFloat(document.getElementById("ymax").value);
		var xMin = parseFloat(document.getElementById("xmin").value);
		var yMin = parseFloat(document.getElementById("ymin").value);
		var zMax = parseFloat(document.getElementById("zmax").value);
		var zMin = parseFloat(document.getElementById("zmin").value);
		var startx = parseFloat(document.getElementById("startx").value);
		var starty = parseFloat(document.getElementById("starty").value);
		var startz = parseFloat(document.getElementById("startz").value);
		var endx = parseFloat(document.getElementById("endx").value);
		var endy = parseFloat(document.getElementById("endy").value);
		var endz = parseFloat(document.getElementById("endz").value);
		graphData.set("expr",
				function (emit, x, y) {
					emit(x, zFunc(x, y), y);
				}
		);
		if (zMax != 0 || zMin!=0){
			zAutofit=false;
		}
		if (zAutofit) {
			var xStep = (xMax - xMin) / 256;
			var yStep = (yMax - yMin) / 256;
			var zSmallest = zFunc(xMin, yMin);
			var zBiggest = zFunc(xMin, yMin);
			for (var x = xMin; x <= xMax; x += xStep) {
				for (var y = yMin; y <= yMax; y += yStep) {
					var z = zFunc(x, y);
					if (z < zSmallest) zSmallest = z;
					if (z > zBiggest) zBiggest = z;
				}
			}
			zMin = zSmallest;
			zMax = zBiggest;
		}
		view.set("range", [[xMin, xMax], [zMin, zMax], [yMin, yMax]]);

		if (graphColorStyle == "grayscale") {
			// zMax = white, zMin = black
			graphColors.set("expr",
					function (emit, x, y) {
						var z = zFunc(x, y);
						var percent = (z - zMin) / (zMax - zMin);
						emit(percent, percent, percent, 1.0);
					}
			);
		} else if (graphColorStyle == "rainbow") {
			// rainbow hue; zMax = red, zMin = violet
			graphColors.set("expr",
					function (emit, x, y) {
						var z = zFunc(x, y);
						var percent = (z - 1.2 * zMin) / (zMax - 1.2 * zMin);
						var color = new THREE.Color(0xffffff);
						color.setHSL(1 - percent, 1, 0.5);
						emit(color.r, color.g, color.b, 1.0);
					}
			);
		} else if (graphColorStyle == "solid blue") {
			// just a solid blue color
			graphColors.set("expr",
					function (emit) {
						emit(0.5, 0.5, 1.0, 1.0);
					}
			);
		} else if (graphColorStyle == "rb") {
			// rainbow hue; zMax = red, zMin = violet
			graphColors.set("expr",
					function (emit, x, y) {
						var z = zFunc(x, y);
						var percent = (z - 0.9 * zMin) / (zMax - 0.9 * zMin);
						var color = new THREE.Color(0xffffff);
						color.setHSL(1 - 0.3 * percent, 1, 0.5);
						emit(color.r, 0.0, color.b, 1.0);
					}
			);
		}
		if (wireframe == "true") {
			graphViewWire.set("visible", true);
		} else {
			graphViewWire.set("visible", false);
		}
		if (shadow == "true") {
			graphViewSolid.set("shaded", true);
		} else {
			graphViewSolid.set("shaded", false);
		}

		if(points.length!=0){
				var start = points[currVecNum].split(',');
				var end = points[currVecNum+1].split(',');
				var vecData =  view.array({
					width: 1, items: 2, channels: 3,
					data: [ [parseFloat(start[0]),parseFloat(start[2]),parseFloat(start[1])],[parseFloat(end[0]), parseFloat(end[2]), parseFloat(end[1])] ],
				});
		}else {
			var vecData = view.array({
				width: 1, items: 2, channels: 3,
				data: [ [startx, startz, starty],[endx, endz, endy] ],
			});
		}
		vec.set("points", vecData);
	}

	// end of updateGraph function ==============================================================


	var updateGraph = function() { updateGraphFunc(); };
	var NextVec= function(){
		if(points.length== currVecNum+2)
			window.alert("No more steps");
		else{
			currVecNum++;
			var start = points[currVecNum].split(',');
			var end = points[currVecNum+1].split(',');
			var vecData = view.array({
				width: 1, items: 2, channels: 3,
				data: [ [parseFloat(start[0]),parseFloat(start[2]),parseFloat(start[1])],[parseFloat(end[0]), parseFloat(end[2]), parseFloat(end[1])] ],
			});
			vec.set("points", vecData);
			if (displayTrace){
				start = points[currVecNum-1].split(',');
				end = points[currVecNum].split(',');
				vecData = view.array({
					width: 1, items: 2, channels: 3,
					data: [ [parseFloat(start[0]),parseFloat(start[2]),parseFloat(start[1])],[parseFloat(end[0]), parseFloat(end[2]), parseFloat(end[1])] ],
				});
				trace.push(view.vector({
					color: "black", width: 8, end: true, visible: true, points: vecData
				}));
				displayTrace = !displayTrace;
			}else{
				displayTrace = !displayTrace;
			}
		}
	};
	var PrevVec= function(){
		if(currVecNum== 0)
			window.alert("This is the first step");
		else{
			currVecNum--;
			var start = points[currVecNum].split(',');
			var end = points[currVecNum+1].split(',');
			var vecData = view.array({
				width: 1, items: 2, channels: 3,
				data: [ [parseFloat(start[0]),parseFloat(start[2]),parseFloat(start[1])],[parseFloat(end[0]), parseFloat(end[2]), parseFloat(end[1])] ],
			});
			vec.set("points", vecData);
			if (!displayTrace){
				trace.pop().remove();
				displayTrace = !displayTrace;
			}else{
				displayTrace = !displayTrace;
			}
		}
	};
	// setting proxy:true allows interactive controls to override base position
	var camera = mathbox.camera( { proxy: true, position: [4,2,4] } );

	 // save as variable to adjust later
    view = mathbox.cartesian(
	  {
        range: [[xMin, xMax], [yMin, yMax], [zMin,zMax]],
        scale: [2,1,2],
      }
	);

	// axes
	var xAxis = view.axis( {axis: 1, width: 8, detail: 40, color:"red"} );
    var xScale = view.scale( {axis: 1, divide: 10, nice:true, zero:true} );
    var xTicks = view.ticks( {width: 5, size: 15, color: "red", zBias:2} );
    var xFormat = view.format( {digits: 2, font:"Arial", weight: "bold", style: "normal", source: xScale} );
    var xTicksLabel = view.label( {color: "red", zIndex: 0, offset:[0,-20], points: xScale, text: xFormat} );

	var yAxis = view.axis( {axis: 3, width: 8, detail: 40, color:"green"} );
    var yScale = view.scale( {axis: 3, divide: 5, nice:true, zero:false} );
    var yTicks = view.ticks( {width: 5, size: 15, color: "green", zBias:2} );
    var yFormat = view.format( {digits: 2, font:"Arial", weight: "bold", style: "normal", source: yScale} );
    var yTicksLabel = view.label( {color: "green", zIndex: 0, offset:[0,0], points: yScale, text: yFormat} );

	var zAxis = view.axis( {axis: 2, width: 8, detail: 40, color:"blue"} );
    var zScale = view.scale( {axis: 2, divide: 5, nice:true, zero:false} );
    var zTicks = view.ticks( {width: 5, size: 15, color: "blue", zBias:2} );
    var zFormat = view.format( {digits: 2, font:"Arial", weight: "bold", style: "normal", source: zScale} );
    var zTicksLabel = view.label( {color: "blue", zIndex: 0, offset:[0,0], points: zScale, text: zFormat} );

	view.grid( {axes:[1,3], width: 2, divideX: 20, divideY: 20, opacity:0.25} );


	var graphData = view.area({
		axes: [1,3], channels: 3, width: 64, height: 64,
        expr: function (emit, x, y)
		{
		  var z = x*y;
          emit( x, z, y );
        },
    });

	// actuall emitter set later.
	var graphColors = view.area({
		expr: function (emit, x)
		{
			if (x < 0)
				emit(1.0, 0.0, 0.0, 1.0);
		    else
				emit(0.0, 1.0, 0.0, 1.0);
		},
		axes: [1,3],
		width:  64, height: 64,
		channels: 4, // RGBA
    });
	var vecData = view.array({
		width: 1, items: 2, channels: 3,
		data: [ [0,0,0],[1,1,1] ],
	});
	var vec = view.vector({
		color: "orange", width: 8, end: true, visible: true, points: vecData
 	 });

	var  trace =[];

	// create graph in two parts, because want solid and wireframe to be different colors
	// shaded:false for a solid color (curve appearance provided by mesh)
	// width: width of line mesh
	// note: colors will mult. against color value, so set color to white (#FFFFFF) to let colors have complete control.
	var graphShaded = false;
	var graphViewSolid = view.surface({
		points:graphData,
		color:"#FFFFFF", shaded:false, fill:true, lineX:false, lineY:false, colors:graphColors, visible:true, width:0
	});

	var graphWireVisible = true;
	var graphViewWire = view.surface({
		points: graphData,
		color:"#000000", shaded:false, fill:false, lineX:true, lineY:true, visible:graphWireVisible, width:2
    });
	updateGraphFunc();
	</script>
</body>
</html>
