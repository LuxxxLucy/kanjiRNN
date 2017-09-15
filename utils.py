import os
import cPickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import svgwrite
from IPython.display import SVG, display
from svg.path import Path, Line, Arc, CubicBezier, QuadraticBezier, parse_path

def draw_sketch_array(strokes_array, svg_only = False):
  draw_stroke_color_array(strokes_array, factor=sample_args.scale_factor, maxcol = sample_args.num_col, svg_filename = sample_args.filename+'.svg', stroke_width = sample_args.stroke_width, block_size = sample_args.picture_size, svg_only = svg_only, color_mode = color_mode)

def calculate_start_point(data, factor=1.0, block_size = 200):
	# will try to center the sketch to the middle of the block
	# determines maxx, minx, maxy, miny
	sx = 0
	sy = 0
	maxx = 0
	minx = 0
	maxy = 0
	miny = 0
	for i in range(len(data)):
		sx += round(float(data[i, 0])*factor, 3)
		sy += round(float(data[i, 1])*factor, 3)
		maxx = max(maxx, sx)
		minx = min(minx, sx)
		maxy = max(maxy, sy)
		miny = min(miny, sy)

	abs_x = block_size/2-(maxx-minx)/2-minx
	abs_y = block_size/2-(maxy-miny)/2-miny

	return abs_x, abs_y, (maxx-minx), (maxy-miny)

def draw_stroke_color_array(data, factor=1, svg_filename = 'sample.svg', stroke_width = 1, block_size = 200, maxcol = 5, svg_only = False, color_mode = True):

	num_char = len(data)

	if num_char < 1:
		return

	max_color_intensity = 225

	numrow = np.ceil(float(num_char)/float(maxcol))
	dwg = svgwrite.Drawing(svg_filename, size=(block_size*(min(num_char, maxcol)), block_size*numrow))
	dwg.add(dwg.rect(insert=(0, 0), size=(block_size*(min(num_char, maxcol)), block_size*numrow),fill='white'))

	the_color = "rgb("+str(random.randint(0, max_color_intensity))+","+str(int(random.randint(0, max_color_intensity)))+","+str(int(random.randint(0, max_color_intensity)))+")"

	for j in range(len(data)):

		lift_pen = 0
		#end_of_char = 0
		cdata = data[j]
		abs_x, abs_y, size_x, size_y = calculate_start_point(cdata, factor, block_size)
		abs_x += (j % maxcol) * block_size
		abs_y += (j / maxcol) * block_size

		for i in range(len(cdata)):

			x = round(float(cdata[i,0])*factor, 3)
			y = round(float(cdata[i,1])*factor, 3)

			prev_x = round(abs_x, 3)
			prev_y = round(abs_y, 3)

			abs_x += x
			abs_y += y

			if (lift_pen == 1):
				p = "M "+str(abs_x)+","+str(abs_y)+" "
				the_color = "rgb("+str(random.randint(0, max_color_intensity))+","+str(int(random.randint(0, max_color_intensity)))+","+str(int(random.randint(0, max_color_intensity)))+")"
			else:
				p = "M "+str(prev_x)+","+str(prev_y)+" L "+str(abs_x)+","+str(abs_y)+" "

			lift_pen = max(cdata[i, 2], cdata[i, 3]) # lift pen if both eos or eoc
			#end_of_char = cdata[i, 3] # not used for now.

			if color_mode == False:
				the_color = "#000"

			dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill(the_color)) #, opacity=round(random.random()*0.5+0.5, 3)

	dwg.save()
	if svg_only == False:
		display(SVG(dwg.tostring()))

def draw_stroke_color(data, factor=1, svg_filename = 'sample.svg', stroke_width = 1, block_size = 200, maxcol = 5, svg_only = False, color_mode = True):

	def split_sketch(data):
		# split a sketch with many eoc into an array of sketches, each with just one eoc at the end.
		# ignores last stub with no eoc.
		counter = 0
		result = []
		for i in range(len(data)):
			eoc = data[i, 3]
			if eoc > 0:
				result.append(data[counter:i+1])
				counter = i+1
		#if (counter < len(data)): # ignore the rest
		#  result.append(data[counter:])
		return result

	data = np.array(data, dtype=np.float32)
	data = split_sketch(data)
	draw_stroke_color_array(data, factor, svg_filename, stroke_width, block_size, maxcol, svg_only, color_mode)
