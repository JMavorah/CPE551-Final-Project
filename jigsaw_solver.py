#Jigsaw Puzzle Solver using Canny edge detection and template matching in OpenCV

#import packages:
import cv2 #OpenCV
import numpy as np
import sys


puzzle_color = cv2.imread("Resources\VehiclesPuzzle\puzzle.jpg") #read in the (color) puzzle image
pieces_color = cv2.imread("Resources\VehiclesPuzzle\ScrambledPieces.png") #read in the (color) pieces image
puzzle_width = 4
puzzle_height = 3
num_pieces = puzzle_width * puzzle_height
templates_color = [None]*(num_pieces)
templates = [None]*(num_pieces)


#function to perform Canny edge detection, locate contours, and extract pieces from image:
def canny_edge():

    global pieces_color
    global templates_color
    global templates
    image = pieces_color.copy() #make copy of puzzle image to modify

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #make image grayscale
    image_blurred = cv2.GaussianBlur(image_gray, (3,3), 0) #blurred image to apply canny 

    image_canny = cv2.Canny(image_blurred, 120, 255, 1) #apply canny edge detection
    kernel = np.ones((5, 5), 'uint8') #kernel for morphological filter

    #apply morphological filter to image to help find shape of pieces
    image_dilated = cv2.dilate(image_canny, kernel, iterations=1) 

    #OpenCV contour detection
    contours = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 2):
        contours = contours[0]
    else:
        contours = contours[1] 

    #now loop through all found contours and extract separate images
    count = 0
    for contour in contours:

        #Place bounding box around contour
        x_loc, y_loc, width, height = cv2.boundingRect(contour)

        piece = image[y_loc:y_loc + height, x_loc:x_loc + width] #extract image of single piece
        cv2.imwrite("Resources\VehiclesPuzzle\Pieces\piece"+str(count)+".png",piece) #write image of piece

        templates[count] = cv2.imread("Resources\VehiclesPuzzle\Pieces\piece"+str(count)+".png",0) #add piece to templates
        templates_color[count] = cv2.imread("Resources\VehiclesPuzzle\Pieces\piece"+str(count)+".png",-1)

        ##---
        #draw rectangle around piece
        cv2.rectangle(image, (x_loc,y_loc), (x_loc + width, y_loc + height), (0,0,255), 2)

        count+=1 #increment count of pieces

    cv2.imshow("Canny Edge Detection", image_canny)
    cv2.imshow("Pieces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#function to perform template matching to find correct piece placement:
def template_match(image, template, count):

    #height of template image
    height, width = template.shape
    results = [None] * 6

    #template matching formulas included in OpenCV:
    TM_methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF]

    #perform template matching with each method:
    for method in TM_methods:

        puzzle_color = image.copy() #make copy of puzzle image and convert to grayscale
        puzzle = cv2.cvtColor(puzzle_color, cv2.COLOR_BGR2GRAY)

        #perform template matching and get location and size of bounding box
        result = cv2.matchTemplate(puzzle, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        #location is returned differently for these two methods:
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else: location = max_loc

        rect_bot_right = (location[0] + width, location[1] + height)
        color = (0,0,255)
        cv2.imshow('Template',templates_color[count])

        #put bounding box around correct placement
        cv2.rectangle(puzzle_color, location, rect_bot_right, color, 3)
        cv2.imshow('Match', puzzle_color)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #return proper location of puzzle piece
        if method in [cv2.TM_CCOEFF_NORMED]:
            return(location)

#function to build and display the solution of the puzzle: 
def construct_puzzle(locations):

    #np array of piece indices (currently ordered 0-11):
    pieces = [None] * puzzle_width*puzzle_height
    for i in range(puzzle_width*puzzle_height):
        pieces[i] = i
    pieces = np.asarray(pieces)

    #list of puzzle piece filenames:
    puzzle_pieces = [None] * num_pieces
    for piece in range(num_pieces):
        puzzle_pieces[piece] = cv2.imread("Resources\VehiclesPuzzle\Pieces\piece" + str(piece) + ".png")
        puzzle_pieces[piece] = cv2.resize(puzzle_pieces[piece], (150, 170))

    #convert location to np array:
    locations = locations
    np_locations = np.asarray(locations)

    #sort locations by y, then x (arrange left to right, top to bottom):
    ind = np.lexsort((np_locations[:,0],np_locations[:,1]))

    #normalize into proper number of rows to account for variations in y value:
    np_locations_norm = np_locations[ind]

    row = 0
    count = 0
    for i in range(len(np_locations)):
        np_locations_norm[i][1] = row
        #print(np_locations_norm[i][1])
        count += 1
        if(count == puzzle_width): #end of row, go to next
            row += 1
            count = 0

    #sort again into final order
    piece_order = np.lexsort((np_locations_norm[:,0],np_locations_norm[:,1]))
    np_locations_norm_sort = np_locations_norm[piece_order]

    #apply indices from both previous sorts to get final placement order of pieces:
    placement_order = pieces[ind][piece_order]

    #finally, place the pieces and display solution:
    row_image = np.zeros((170,1,3), 'uint8') #to build the rows of the puzzles
    full_image = np.zeros((1,601,3), 'uint8') # to build the completed puzzle
    
    piece_counter = 0

    #iterate through rows and columns of puzzle
    for row in range(puzzle_height):
        for col in range(puzzle_width):
            #build row of puzzle
            row_image = np.concatenate((row_image, puzzle_pieces[placement_order[piece_counter]]), axis = 1) 
            piece_counter += 1

        #add row to full image, and reset row to to fill the next row:
        full_image = np.concatenate((full_image, row_image), axis = 0)
        row_image = np.zeros((170,1,3), 'uint8')
    
    #Show final image:
    cv2.imshow("Solution", full_image)
    cv2.waitKey(0)

    

def main():

    locations = [None] * (num_pieces) #return locations of pieces from template matching
    
    
    canny_edge() #apply the Canny edge detection and find contours to extract the pieces

    #perform template matching on each piece:
    count = 0
    for template in templates:
        result = template_match(puzzle_color,template,count)
        locations[count] = result
        count += 1
     
    construct_puzzle(locations) #construct and display the puzzle


if __name__ == "__main__":
    main() #run the main function

