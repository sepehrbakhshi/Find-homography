import numpy as np
import cv2 as cv
from sklearn import preprocessing
import scipy
import sys
import time

def get_descriptor_keypoint(img_name,sift_keypoints):
    img_1 = cv.imread(img_name)
    gray_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp_1 = sift.detect(gray_1,None)
    img=cv.drawKeypoints(gray_1,kp_1,img_1)
    cv.imwrite(sift_keypoints,img_1)

    img=cv.drawKeypoints(gray_1,kp_1,img_1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(sift_keypoints,img_1)

    sift_1 = cv.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift_1.detectAndCompute(gray_1,None)
    kp_1 = np.array(kp_1)
    des_1 = np.array(des_1)
    return kp_1 , des_1

def normalizer(des_1):
    normalized_descriptors = []
    for i in range(0 , des_1.shape[0]):
        X = des_1[i]
        X = X.reshape(-1,128)
        #normalize data to [0,1]
        des_norm = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=1, copy=True)
        #clip values to 0.2
        des_norm = np.clip(des_norm, a_min = 0, a_max = 0.2)
        normalized_descriptors.append(des_norm)
    normalized_descriptors = np.array(normalized_descriptors)
    normalized_descriptors = normalized_descriptors.reshape(normalized_descriptors.shape[0] , -1)
    return normalized_descriptors

def get_nndr(normalized_descriptor_1,normalized_descriptor_2):
    #KDTree
    my_kdd_im1 = scipy.spatial.cKDTree(normalized_descriptor_2, leafsize=200)
    neighbours = []
    new_kp_list = []
    for item in normalized_descriptor_1:
        neighbour_tuples = my_kdd_im1.query(item, k=2, distance_upper_bound=6)
        neighbours.append(neighbour_tuples)
    #a 3d array : the first dimension is the id of descriptor, the second dimension is
    #distances to the nearest neighbours and the 3rd dimension is the id of neighbours
    neighbours = np.array(neighbours)

    nndr = []
    point_numbers = []
    for i in range(0 , neighbours.shape[0]):
        nndr_results = neighbours[i][0][0]/neighbours[i][0][1]

        #points = np.zeros((1,2))
        point = neighbours[i][1][0]
        #points[0,1] = neighbours[i][1][1]
        point_numbers.append(point)
        nndr.append(nndr_results)
    nndr = np.array(nndr)
    point_numbers = np.array(point_numbers)
    return nndr , point_numbers
def get_best_points(nndr,threshold):
    counter = 0
    new_point_ids = []
    for i in range(0 , nndr.shape[0]):
        if(nndr[i] < threshold):
            tmp_points = np.zeros((1,2))
            tmp_points[0,0] = i
            tmp_points[0,1] = points[i]
            new_point_ids.append(tmp_points)
            counter += 1

    new_point_ids = np.array(new_point_ids)
    new_point_ids = new_point_ids.reshape(new_point_ids.shape[0] , -1)
    return new_point_ids
def draw_matchedpoints(im1_name,im2_name,new_point_ids):

    img1 = cv.imread(im1_name)
    img2 = cv.imread(im2_name)
    height = img1.shape[0]
    width = img2.shape[1]
    img2 = cv.resize(img2, (width, height))
    vis = np.concatenate((img1, img2), axis=1)
    for i in range(0 , len(new_point_ids)):

        xy = kp_1[int(new_point_ids[i,0])].pt
        xy_1 = (int(xy[0]) , int(xy[1]) )
        #xy_2 = (int(xy[0])+3 , int(xy[1])+3 )
        xy = kp_2[int(new_point_ids[i,1])].pt
        xy_2 = (int(xy[0])+width , int(xy[1]) )
        #cv.imwrite('out.png', vis)
        cv.line(vis, xy_1, xy_2, (0,204,0), 1)
    imS_1 = cv.resize(vis, (1260, 540))
    cv.imwrite('matched_points.png', imS_1)
    cv.imshow("Image", imS_1)
    cv.waitKey(0)
    cv.destroyAllWindows()

def jacobian(px_1 , py_1 , x_g_1, y_g_1):
    tmp_j = ([[x_g_1 , y_g_1 , 1 , 0, 0, 0, -((px_1*x_g_1)) , -((px_1 * y_g_1))], [ 0, 0, 0 , x_g_1 , y_g_1 , 1, -((x_g_1 * py_1)) , -((py_1*y_g_1))]])
    tmp_j = np.array(tmp_j)
    j_1 = tmp_j
    return j_1

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

def find_homography(four_points,kp_1,kp_2):

    points = np.zeros((four_points.shape[0],2))
    points_prim = np.zeros((four_points.shape[0],2))

    for i in range(0 , points.shape[0]):
        loc_1 = int(four_points[i,0])
        p_1 = kp_1[loc_1].pt
        px_1 = int(p_1[0])
        py_1 = int(p_1[1])
        points[i,0] = px_1
        points[i,1] = py_1
    for i in range(0 , points_prim.shape[0]):
        loc_prim_1 = int(four_points[i,1])
        p_prim_1 = kp_2[loc_prim_1].pt
        px_primt_1 = int(p_prim_1[0])
        py_primt_1 = int(p_prim_1[1])
        points_prim[i,0] = px_primt_1
        points_prim[i,1] = py_primt_1

    

    tmp = points
    points = points_prim
    points_prim = tmp

    my_jacobian = []
    for i in range(0 , points_prim.shape[0]):
        j_1 = jacobian(points[i,0] , points[i,1] , points_prim[i,0], points_prim[i,1])
        my_jacobian.append(j_1)
    my_jacobian = np.array(my_jacobian)
    my_jacobian = my_jacobian.reshape((my_jacobian.shape[0] * my_jacobian.shape[1],-1))

    reg = 2**-30
    pseudoinverse = pinv(my_jacobian,reg)

    vector_points = points.ravel()

    homo = np.matmul(pseudoinverse,vector_points)

    homo = homo.reshape((8,1))
    f_homo = []
    for i in range(0 , homo.shape[0]):
        f_homo.append(float(homo[i]))
    f_homo.append(1)
    f_homo = np.array(f_homo)
    f_homo = f_homo.reshape((3,3))
    print("##########")
    print(f_homo)
    print("##########")


    return f_homo

    """

    hhhh = cv.findHomography(points, points_prim)
    hhhh = np.array(hhhh)
    print('###################')
    print(hhhh[0])
    in_pts = np.matmul(hhhh[0],[px_1 , py_1,1])
    print(in_pts)
    print(px_primt_1, py_primt_1)
    #print(hhhh[0])
    #print(cv.findHomography(points_prim, points))
    """
def my_RANSAC(new_point_ids,kp_1,kp_2,n_iterations,diff_threshold, numberof_starting_points,percentage_of_datapoints):
    #numberof_starting_points : 8 points are chosen for detecting the homography
    #percentage_of_datapoints : 50% of data is chosen to find the outliers and inliers
    #diff_threshold : threshold for choosing the inlier
    # number of iterations to choose the inliers
    counter = 0
    prev_inlier_count = 0
    final_inliers = []
    final_outliers = []
    final_starting_points = []
    min_differences = []
    final_homo =[]
    while counter < n_iterations:

        n = new_point_ids.shape[0]
        #index = np.random.choice(new_point_ids.shape[0], int((n/2)) , replace=False)
        #random_inliers = new_point_ids[index]
        index = np.random.choice(new_point_ids.shape[0], numberof_starting_points , replace=False)
        starting_points = new_point_ids[index]
        other_points = np.delete(new_point_ids, index , 0)

        first_image_points = other_points[:,0]
        second_image_points = other_points[:,1]
        first_image_locations = []
        second_image_locations = []
        first_image_points = np.array(first_image_points)
        second_image_points = np.array(second_image_points)

        first_image_points_xy = np.zeros((first_image_points.shape[0],2))
        for i in range(0 , first_image_points.shape[0]):
            loc_1 = int(first_image_points[i])
            p_1 = kp_1[loc_1].pt
            px_1 = int(p_1[0])
            py_1 = int(p_1[1])
            first_image_points_xy[i,0] = px_1
            first_image_points_xy[i,1] = py_1
            first_image_locations.append(loc_1)

        second_image_points_xy = np.zeros((second_image_points.shape[0],2))
        for i in range(0 , second_image_points.shape[0]):
            loc_1 = int(second_image_points[i])
            p_1 = kp_2[loc_1].pt
            px_1 = int(p_1[0])
            py_1 = int(p_1[1])
            second_image_points_xy[i,0] = px_1
            second_image_points_xy[i,1] = py_1
            second_image_locations.append(loc_1)

        tmp_homo = find_homography(starting_points,kp_1,kp_2)
        homogenoues_ones = np.ones((first_image_points.shape[0],1))

        homogenoues_first_image = np.concatenate((first_image_points_xy,homogenoues_ones) , axis =1 )

        guessed_output = np.matmul(tmp_homo , homogenoues_first_image.T)

        homogenous_coeff = guessed_output[2,:]
        guessed_output = guessed_output / homogenous_coeff
        guessed_output = guessed_output.T
        guessed_output = guessed_output[:,0:2]

        differences = np.linalg.norm(guessed_output - second_image_points_xy,axis=1)
        #print differences to see how good does the linear homography works
        differences = np.array(differences)

        inliers = []
        outliers = []
        for i in range(0 , differences.shape[0]):
            if(differences[i] < diff_threshold):
                tmp = np.zeros((1,2))
                tmp[0,0] = first_image_locations[i]
                tmp[0,1] = second_image_locations[i]
                inliers.append(tmp)
            else:
                tmp = np.zeros((1,2))
                tmp[0,0] = first_image_locations[i]
                tmp[0,1] = second_image_locations[i]
                outliers.append(tmp)
        if((len(inliers) > percentage_of_datapoints * new_point_ids.shape[0]) and len(inliers) > prev_inlier_count):
            final_inliers = inliers
            final_outliers = outliers
            final_starting_points = starting_points
            prev_inlier_count = len(inliers)
            min_differences = differences
            final_homo = tmp_homo
        counter += 1

    final_inliers = np.array(final_inliers)
    final_inliers = final_inliers.reshape((final_inliers.shape[0] , -1))
    final_outliers = np.array(final_outliers)
    final_outliers = final_outliers.reshape((final_outliers.shape[0] , -1))
    final_starting_points = np.array(final_starting_points)

    ##############
    #Here is the code which shows the effectiveness of the homography which is calculated linearly.
    #we can use this part in nonlinear estimation too (if np.linalg.norm(dif) < dif_threshold:)
    ##############
    homogenoues_ones = np.ones((first_image_points.shape[0],1))

    homogenoues_first_image = np.concatenate((first_image_points_xy,homogenoues_ones) , axis =1 )
    guessed_output = np.matmul(final_homo , homogenoues_first_image.T)

    homogenous_coeff = guessed_output[2,:]
    guessed_output = guessed_output / homogenous_coeff
    guessed_output = guessed_output.T
    guessed_output = guessed_output[:,0:2]

    differences = np.linalg.norm(guessed_output - second_image_points_xy,axis=1)
    final_differences = differences


    return final_inliers , final_outliers , final_starting_points , prev_inlier_count , final_homo , final_differences
    #we should calculate the homography outside of the model.

def compute_nonlinear_jacobian(homo,first_image_points_xy):

    total_jacobian = []
    total_jacobian_x = []
    total_jacobian_y = []
    jacobian_x = np.zeros((9,1))
    jacobian_y = np.zeros((9,1))
    for i in range(first_image_points_xy.shape[0]):
        #calculate for x and y
        tmp_x = first_image_points_xy[i,0]
        tmp_y = first_image_points_xy[i,1]

        nominator = (homo[0,0] * tmp_x) + (homo[1,1] * tmp_y) + homo[1,2]
        denominator = (homo[2,0] * tmp_x) + (homo[2,1] * tmp_y) + homo[2,2]
        #calculating the derivatives
        jacobian_x[0] = -1 * tmp_x/denominator
        jacobian_x[1] = -1 * tmp_y/denominator
        jacobian_x[2] = -1/denominator
        jacobian_x[6] = float((nominator * tmp_x)/(denominator ** 2))
        jacobian_x[7] = (nominator * tmp_y)/(denominator ** 2)
        jacobian_x[8] = nominator/(denominator ** 2)

        jacobian_y[3] = (-1 * tmp_x) / denominator
        jacobian_y[4] =( -1 * tmp_y )/ denominator
        jacobian_y[5] = -1 / denominator
        jacobian_y[6] = (nominator * tmp_x)/(denominator ** 2)
        jacobian_y[7] = (nominator * tmp_y)/(denominator ** 2)
        jacobian_y[8] = nominator/(denominator ** 2)

        total_jacobian.append(jacobian_x)
        total_jacobian.append(jacobian_y)

    total_jacobian = np.array(total_jacobian)
    total_jacobian = total_jacobian.reshape((total_jacobian.shape[0] , -1))
    return total_jacobian
def calculate_residual(homo,x,x_prim):

    homogenoues_ones = np.ones((x.shape[0],1))
    homogenoues_first_image = np.concatenate((x,homogenoues_ones) , axis =1 )
    guessed_output = np.matmul(homo , homogenoues_first_image.T)
    guessed_output = np.matmul(homo , homogenoues_first_image.T)

    homogenous_coeff = guessed_output[2,:]
    guessed_output = guessed_output / homogenous_coeff
    guessed_output = guessed_output.T
    guessed_output = guessed_output[:,0:2]

    residual = x_prim - guessed_output

    residual = residual.flatten()
    residual = np.array(residual)
    residual = residual.reshape((residual.shape[0] ,  1))
    return residual

def fit_non_linear_leastsquare(homo,inliers,starting_points,dif_threshold,iterations):

    all_inliers = np.concatenate((inliers,starting_points))

    first_image_points = all_inliers[:,0]
    second_image_points = all_inliers[:,1]
    first_image_locations = []
    second_image_locations = []
    first_image_points = np.array(first_image_points)
    second_image_points = np.array(second_image_points)

    first_image_points_xy = np.zeros((first_image_points.shape[0],2))
    for i in range(0 , first_image_points.shape[0]):
        loc_1 = int(first_image_points[i])
        p_1 = kp_1[loc_1].pt
        px_1 = int(p_1[0])
        py_1 = int(p_1[1])
        first_image_points_xy[i,0] = px_1
        first_image_points_xy[i,1] = py_1
        first_image_locations.append(loc_1)

    second_image_points_xy = np.zeros((second_image_points.shape[0],2))
    for i in range(0 , second_image_points.shape[0]):
        loc_1 = int(second_image_points[i])
        p_1 = kp_2[loc_1].pt
        px_1 = int(p_1[0])
        py_1 = int(p_1[1])
        second_image_points_xy[i,0] = px_1
        second_image_points_xy[i,1] = py_1
        second_image_locations.append(loc_1)

    #calculate jacobian
    total_jacobian = compute_nonlinear_jacobian(homo,first_image_points_xy)
    beta_prev = 0.8 * np.amax(np.diag(total_jacobian))
    h_prev = homo

    residual_prev = calculate_residual(h_prev,first_image_points_xy,second_image_points_xy)
    cost_prev = np.dot(residual_prev.T, residual_prev)
    counter = 0
    final_result = []
    while counter < iterations:

        total_jacobian = compute_nonlinear_jacobian(h_prev, first_image_points_xy)

        dif = np.linalg.inv(np.dot(total_jacobian.T, total_jacobian) + beta_prev * np.eye(total_jacobian.shape[1], total_jacobian.shape[1]))
        jacob_t = (-1 * total_jacobian.T)
        dif = np.dot(np.dot(dif, jacob_t) , residual_prev)

        if np.linalg.norm(dif) < dif_threshold:
            final_result = h_prev

        dif_3by3 = dif.copy()
        dif_3by3 = dif_3by3.reshape((3,3))
        h_new = h_prev + dif_3by3


        residual_new = calculate_residual(h_new, first_image_points_xy, second_image_points_xy)

        cost_new = np.dot(residual_new.T, residual_new)

        cost_difs = (cost_prev - cost_new)
        new_difs = np.dot(np.dot(dif.T, -1*total_jacobian.T), residual_prev)
        new_difs = new_difs + np.dot(np.dot(dif.T, beta_prev * np.eye(total_jacobian.shape[1], total_jacobian.shape[1])), dif)
        rho = cost_difs/new_difs

        beta_prev = beta_prev * max(1/3, 1 - (2 * rho - 1)**3)

        if cost_new < cost_prev:
            h_prev = h_new
            residual_prev = residual_new
            cost_prev = cost_new

        #if the difference is not less than threshold then continue and update the values
        counter += 1

#################################################################################
#################################################################################
######################MAIN PART OF THE CODE BEGINS ##############################
#################################################################################
#################################################################################
np.set_printoptions(threshold=sys.maxsize)

#First image
kp_1 , des_1 = get_descriptor_keypoint('im1.png','sift_keypoints_1.jpg')
normalized_descriptors_1 = normalizer(des_1)
#nndr_1 = get_nndr(normalized_descriptors_1)
des_size_1 = len(des_1)
#Second image
kp_2 , des_2 = get_descriptor_keypoint('im2.png','sift_keypoints_2.jpg')
normalized_descriptors_2 = normalizer(des_2)
#nndr_2 = get_nndr(normalized_descriptors_2)
des_size_2 = len(des_2)
#normalized_descriptors_all = np.concatenate((normalized_descriptors_1, normalized_descriptors_2))
#kp_all = np.concatenate((kp_1, kp_2))
nndr , points= get_nndr(normalized_descriptors_1,normalized_descriptors_2)

threshold = 0.7
new_point_ids = get_best_points(nndr,threshold)

percentage_of_datapoints = 0.5
numberof_starting_points = 4
diff_threshold = 2
n_iterations = 100
final_inliers ,final_outliers ,final_starting_points ,prev_inlier_count, final_homo , final_differences= my_RANSAC(new_point_ids,kp_1,kp_2,n_iterations,diff_threshold, numberof_starting_points,percentage_of_datapoints)
#dif_threshold = 10**-6

print(final_homo)

dif_threshold = 10**-6
iterations = 100
final_homography = fit_non_linear_leastsquare(final_homo,final_inliers,final_starting_points,dif_threshold,iterations)

print(final_homography)

img_1 = cv.imread('im1.png')

img_2 = cv.imread('im2.png')
im_2_coordinates = []
for i in range(0,img_2.shape[0]):
    for j in range(0 , img_2.shape[1]):
        xyw = np.zeros((3,1))
        xyw[0] = i
        xyw[1] = j
        xyw[2] = 1
        im_2_coordinates.append(xyw)
im_2_coordinates = np.array(im_2_coordinates)
im_2_coordinates = im_2_coordinates.reshape(im_2_coordinates.shape[0] , -1)
print(im_2_coordinates.shape)
"""
final_homo_inverse = np.linalg.inv(final_homography)
guessed_output = np.matmul(final_homo_inverse , im_2_coordinates.T)
print(guessed_output)

np.save("im2guessed.npy",guessed_output)

"""
guessed_output = np.load("im2guessed.npy")

homogenous_coeff = guessed_output[2,:]
guessed_output = guessed_output / homogenous_coeff
guessed_output = guessed_output.T
guessed_output = guessed_output[:,0:2]
#guessed_output = np.around (guessed_output)
minimums = np.amin(guessed_output, axis=0)
maximums = np.amax(guessed_output, axis=0)
print(minimums)
print(maximums)
print(guessed_output.shape)
print(img_2.shape)
print(img_1.shape)

extra_pixels = np.zeros((1,1065, 3))
img_1 = np.concatenate((img_1,extra_pixels))
print(img_1.shape)

guessed_output[:,1] = guessed_output[:,1] + 160
guessed_output = np.around (guessed_output)
minimums = np.amin(guessed_output, axis=0)
maximums = np.amax(guessed_output, axis=0)
print(minimums)
print(maximums)

my_image = np.zeros((1500,1800,3),np.uint8)

my_image.fill(0)
#my_image = my_image[my_image>=0]=[255,255,255]
my_image = my_image.astype(np.uint8)
image_1_flattened = img_1.copy()
image_1_flattened = image_1_flattened.reshape((-1,3))
print(image_1_flattened.shape)
unq, count = np.unique(guessed_output, axis=0, return_counts=True)
non_unique_values = unq[count>1]
non_unique_values = np.array(non_unique_values)
print(non_unique_values.shape)
print(guessed_output.shape)


for i in range(0,guessed_output.shape[0]):
    #print(int(guessed_output[i,0]))
    #print(int(guessed_output[i,1]))
    print(i)
    #print("$$$$$")
    #print(img_1[i])

    #print(int(guessed_output[i,0]),int(guessed_output[i,1]))

    my_image[int(guessed_output[i,0]),int(guessed_output[i,1])] = image_1_flattened[i]
    #print(my_image[int(guessed_output[i,0]),int(guessed_output[i,1])])


#from PIL import Image
#img = Image.fromarray(my_image.astype(int), 'RGB')

#print(my_image[int(guessed_output[0,0]),int(guessed_output[0,1])])

#my_image = my_image.astype(np.uint8)

#calculating empty pixels
"""
extra = []
for i in range(559 , 1434):
    for j in range(10,1202):
        vv = [i,j]
        vv = np.array(vv)
        #print((guessed_output == vv).all(1).any())


        if(not (guessed_output == vv).all(1).any() ):
            extra.append([i,j])
            print([i,j])
            #print([i,j])(A==B).all()
        elif((my_image[int(guessed_output[i,0]),int(guessed_output[i,1])] == [0,0,0]).all()):
            print([i,j])
            time.sleep(2)
extra = np.array(extra)

print(extra.shape)
np.save("empty_pixels_2.npy",extra)
adasd
"""
extra = np.load("empty_pixels_2.npy")

for i in range(0 , extra.shape[0]):

    #x = np.floor((extra[i,0] - 562) * (800/873))
    #y = np.floor((extra[i,1] - 13) * (1065/1192))


    x = extra[i,0]
    y = extra[i,1]
    x = int(x)
    y = int(y)
    x_1 = x - 1
    y_1 = y - 1
    y_2 = y + 1
    x_2 = x + 1
    print(extra[i,0])
    print(extra[i,1])
    print(x)
    print(y)
    print("-------")
    #time.sleep(2)
    f_q11 = my_image[x_1,y_1]
    f_q12 = my_image[x_1,y_2]
    f_q21 = my_image[x_2,y_1]
    f_q22 = my_image[x_2,y_2]

    f_q11 =  np.array(f_q11)
    f_q12 =  np.array(f_q12)
    f_q21 =  np.array(f_q21)
    f_q22 =  np.array(f_q22)
    p =  ((((x_2 - x )*(y_2 - y))/((x_2 - x_1 )*(y_2 - y_1))) * f_q11) + ((((x - x_1 )*(y_2 - y))/((x_2 - x_1 )*(y_2 - y_1))) * f_q21) +((((x_2 - x )*(y - y_1))/((x_2 - x_1 )*(y_2 - y_1))) * f_q12) + ((((x - x_1 )*(y - y_1))/((x_2 - x_1 )*(y_2 - y_1))) * f_q22)



    #p = f_q11 + f_q12 + f_q21 + f_q22

    print("--------")
    my_image[extra[i,0],extra[i,1]] = p

imS_1 = cv.resize(my_image, (1260, 540))
#imS_1 = cv.resize(my_image, (img_1, 540))
cv.imwrite('out.png', imS_1)
#cv.imshow("Image", imS_1)
img_3 = cv.imread('out.png')
cv.imshow("Image", img_3)
cv.waitKey(0)
cv.destroyAllWindows()
"""

my_image = np.zeros(1564,1153)
for i in range(0,img_1.shape[0]):
    for j in range(88 , img_1.shape[1]+88):
        if(my_image[i,j] != 0):


"""
"""
homogenoues_ones = np.ones((first_image_points.shape[0],1))

homogenoues_first_image = np.concatenate((first_image_points_xy,homogenoues_ones) , axis =1 )
guessed_output = np.matmul(final_homo , homogenoues_first_image.T)

homogenous_coeff = guessed_output[2,:]
guessed_output = guessed_output / homogenous_coeff
guessed_output = guessed_output.T
guessed_output = guessed_output[:,0:2]

differences = np.linalg.norm(guessed_output - second_image_points_xy,axis=1)
final_differences = differences
"""
