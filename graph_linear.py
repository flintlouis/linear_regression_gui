import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import argparse

FPS = 27
GRAPH_SIZE = 2
NAME = "GRAPH"
RIGHT_CLICK = (False, False, True)
LEFT_CLICK = (True, False, False)
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
MAP = 1

iteration = 0

def usage():
	parser = argparse.ArgumentParser(description="Visualization of gradient descent or least squares")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-g", "--gradient", action="store_true", default=True, help="use gradient descent")
	group.add_argument("-l", "--leastsquares", action="store_true", default=False, help="use least squares")
	args = parser.parse_args()
	return args

def draw_line(surface, m=0, b=0):
	x1 = 0
	y1 = m * x1 + b
	x2 = MAP
	y2 = m * x2 + b

	x1 = np.interp(x1, [0, MAP], [0, SCREEN_WIDTH])
	y1 = np.interp(y1, [0, MAP], [SCREEN_HEIGHT, 0])
	x2 = np.interp(x2, [0, MAP], [0, SCREEN_WIDTH])
	y2 = np.interp(y2, [0, MAP], [SCREEN_HEIGHT, 0])

	pygame.draw.line(surface, "red", (x1, y1), (x2, y2), 1)

def plot_data(surface, data):
	if len(data) == 0:
		return
	for point in data:
		x, y = point
		x = np.interp(x, [0, MAP], [0, SCREEN_WIDTH])
		y = np.interp(y, [0, MAP], [SCREEN_HEIGHT, 0])
		pygame.draw.circle(surface, "purple", (x, y), 5)

def least_squares(data):
	np_data = np.array(data)
	X = np_data[:, 0]
	y = np_data[:, 1]
	X_mean = np.mean(X)
	y_mean = np.mean(y)

	numerator, denominator = 0, 0
	numerator = sum((X - X_mean) * (y - y_mean))
	denominator = sum((X - X_mean) ** 2)

	m = numerator / denominator
	b = y_mean - (m * X_mean)
	return m, b

def gradient_descent(data, m, b, learning_rate=0.02):
	global iteration
	iteration += 1
	np_data = np.array(data)
	X = np_data[:, 0]
	y = np_data[:, 1]

	for i in range(len(X)):
		guess = m * X[i] + b
		error = y[i] - guess

		m = m + (error * X[i]) * learning_rate
		b = b + (error) * learning_rate
	return m, b

def handle_keys(data):
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_ESCAPE:
				pygame.quit()
				sys.exit()
		elif event.type == pygame.MOUSEBUTTONDOWN:
			pressed = pygame.mouse.get_pressed()
			x, y = pygame.mouse.get_pos()
			if pressed == LEFT_CLICK:
				x = np.interp(x, [0, SCREEN_WIDTH], [0, MAP])
				y = np.interp(y, [0, SCREEN_HEIGHT], [MAP, 0])
				point = (x, y)
				if point not in data:
					data = data.append(point)
			elif pressed == RIGHT_CLICK:
				pass

def init_pygame():
	pygame.init()
	pygame.display.set_caption(NAME)
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
	surface = pygame.Surface(screen.get_size())
	surface = surface.convert()
	return surface, screen

def display_info(screen, myfont, m, b):
	text = myfont.render(f"m = {m:.5f} b = {b:.5f}", 1, "white")
	screen.blit(text, (10,10))
	text = myfont.render(f"iteration {iteration}", 1, "white")
	screen.blit(text, (10,25))

def load_pygame():
	print("Loading...")
	clock = pygame.time.Clock()
	surface, screen = init_pygame()
	myfont = pygame.font.SysFont("arialblack", 15)
	os.system("clear")
	return clock, surface, screen, myfont

def main():
	args = usage()
	clock, surface, screen, myfont = load_pygame()
	data = []
	m, b = 0, 0
	while True:
		clock.tick(FPS)
		surface.fill("black")
		handle_keys(data)
		plot_data(surface, data)
		if len(data) > 1:
			if args.leastsquares:
				m, b = least_squares(data)
			else:
				m, b = gradient_descent(data, m, b, 0.1)
			draw_line(surface, m, b)
		screen.blit(surface, (0,0))
		display_info(screen, myfont, m, b)
		pygame.display.update()

if __name__ == "__main__":
	main()
