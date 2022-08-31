import numpy as np
import cv2
import matplotlib.pyplot as plt


def accumulator_matrix_creator(edges_matrix):
    """
    This function takes edge matrix and returns accumulator matrix
    :param edges_matrix: A matrix that is non-zero on edge pixels only.
    :return: accumulator matrix, quantized rho and theta vector
    """
    h, w = edges_matrix.shape
    edge_locations = np.argwhere(edges_matrix > 0)
    edge_coordinates = edge_locations  # center = top_left_corner
    theta = np.arange(0, 180, 180 / 1000) / 180 * np.pi
    rho = np.arange(-np.sqrt(w ** 2 + h ** 2), np.sqrt(w ** 2 + h ** 2))
    # print(rho[0])
    accumulator_mat = np.zeros((len(theta), len(rho)), dtype=np.uint8)
    for t in range(len(theta)):
        temp_r = edge_coordinates[:, 1] * np.cos(theta[t]) + edge_coordinates[:, 0] * np.sin(theta[t])
        # print(temp_r)
        temp_r = temp_r - rho[0]
        temp_r = np.round(temp_r).astype(int)
        for r in temp_r:
            if r > accumulator_mat.shape[1]:
                r = accumulator_mat.shape[1] - 1
            elif r < 0:
                r = 0
            accumulator_mat[t, r] += 1

    return accumulator_mat, theta, rho


def lines_polar2mc(accumulator_mat, theta, rho):
    """
    takes the accumulator matrix and quantized theta and rho vectors and finds the lines in m,c coordinate.
    :param accumulator_mat: accumulator matrix
    :param theta: quantized theta vector
    :param rho: quantized rho vector
    :return: vector of m and c of lines
    """
    accumulator_mat_thresholded = accumulator_mat > 160
    lines_ = np.argwhere(accumulator_mat_thresholded)
    m_ = []
    c_ = []
    t_prev = []
    r_prev = []
    vote_prev = []
    for l_ in lines_:
        t_ = theta[l_[0]]
        r_ = rho[l_[1]]
        if np.sin(t_) == 0:
            t_ += 0.0000000001
        flag = True
        for ii in range(len(t_prev)):
            # delete lines that are the same
            if np.abs(t_ - t_prev[ii]) < 20 / 180 * np.pi and np.abs(r_ - r_prev[ii]) < 20:
                if vote_prev[ii] < accumulator_mat[l_[0], l_[1]]:
                    t_prev[ii] = t_
                    r_prev[ii] = r_
                    vote_prev[ii] = accumulator_mat[l_[0], l_[1]]
                    m_[ii] = -np.cos(t_) / np.sin(t_)
                    c_[ii] = r_ / np.sin(t_)
                flag = False
                break

        if flag:
            t_prev.append(t_)
            r_prev.append(r_)
            vote_prev.append(accumulator_mat[l_[0], l_[1]])
            m_.append(-np.cos(t_) / np.sin(t_))
            c_.append(r_ / np.sin(t_))
    return m_, c_


def line_drawer(image, m_vector, c_vector):
    """
    draws lines on given image
    :param image: image
    :param m_vector: vector of slopes
    :param c_vector: vector of intercepts
    :return: image with lines drawn on it
    """
    for i in range(len(m_vector)):
        for j in range(i):
            if np.abs(c_vector[j] - c_vector[i]) < 20:
                continue
        x1, y1 = 0, int(np.round(c_vector[i]))
        x2 = image.shape[1]
        y2 = int(np.round(m_vector[i] * x2 + c_vector[i]))
        image = cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    return image


def find_intersections(image, m, c):
    """
    finds intersection of lines, then chooses valid ones as explained in the report
    :param image: image
    :param m: vector of slopes
    :param c: vector of intercepts
    :return: image with intersection point drawn on it. and valid lines params.
    """
    x_intersection_vec = []
    y_intersection_vec = []
    m1_vec = []
    m2_vec = []
    c1_vec = []
    c2_vec = []
    # finding intersection
    for i in range(len(m)):
        for j in range(i, len(m)):
            if m[i] - m[j] == 0:
                continue
            x_intersection = (c[i] - c[j]) / (m[j] - m[i])
            y_intersection = int(round(m[i] * x_intersection + c[i]))
            x_intersection = int(round(x_intersection))
            if x_intersection < 0 or x_intersection >= image.shape[1]:
                continue
            if y_intersection < 0 or y_intersection >= image.shape[0]:
                continue
            x_intersection_vec.append(x_intersection)
            y_intersection_vec.append(y_intersection)
            m1_vec.append(m[i])
            m2_vec.append(m[j])
            c1_vec.append(c[i])
            c2_vec.append(c[j])

    m_final = []
    c_final = []
    x_intersection_final = []
    y_intersection_final = []
    # Checking if the intersection is valid
    for i in range(len(x_intersection_vec)):
        x = x_intersection_vec[i]
        y = y_intersection_vec[i]
        for j in range(20, 25):
            if y + j >= image.shape[0] or y - j < 0 or x - j < 0 or x + j > image.shape[1]:
                continue
            if np.sum(image[y, x + j, :] > 100) == 3 and np.sum(image[y, x - j, :] > 100) == 3 and \
                    np.sum(image[y - j, x, :] < 60) == 3 and np.sum(image[y + j, x, :] < 60) == 3:
                x_intersection_final.append(x)
                y_intersection_final.append(y)
                m_final.append(m1_vec[i])
                m_final.append(m2_vec[i])
                c_final.append((c1_vec[i]))
                c_final.append((c2_vec[i]))
                break
            elif np.sum(image[y, x + j, :] < 60) == 3 and np.sum(image[y, x - j, :] < 60) == 3 and \
                    np.sum(image[y - j, x, :] > 100) == 3 and np.sum(image[y + j, x, :] > 100) == 3:
                x_intersection_final.append(x)
                y_intersection_final.append(y)
                m_final.append(m1_vec[i])
                m_final.append(m2_vec[i])
                c_final.append((c1_vec[i]))
                c_final.append((c2_vec[i]))
                break
            elif np.sum(image[y, x + j, :] < 60) == 3 and np.sum(image[y, x - j, :] > 100) == 3 and \
                    np.sum(image[y - j, x, :] > 100) == 3 and np.sum(image[y + j, x, :] > 100) == 3:
                x_intersection_final.append(x)
                y_intersection_final.append(y)
                m_final.append(m1_vec[i])
                m_final.append(m2_vec[i])
                c_final.append((c1_vec[i]))
                c_final.append((c2_vec[i]))
                break
            elif np.sum(image[y, x + j, :] > 100) == 3 and np.sum(image[y, x - j, :] < 60) == 3 and \
                    np.sum(image[y - j, x, :] > 100) == 3 and np.sum(image[y + j, x, :] > 100) == 3:
                x_intersection_final.append(x)
                y_intersection_final.append(y)
                m_final.append(m1_vec[i])
                m_final.append(m2_vec[i])
                c_final.append((c1_vec[i]))
                c_final.append((c2_vec[i]))
                break

    m_final_final = []
    c_final_final = []
    for i in range(len(m_final)):
        flag = True
        for j in range(len(m_final_final)):
            if m_final_final[j] == m_final[i] and c_final_final[j] == c_final[i]:
                flag = False
                break
        if flag:
            m_final_final.append(m_final[i])
            c_final_final.append(c_final[i])
    x_intersection_final_final = []
    y_intersection_final_final = []
    # intersecting valid lines again to find the final intersection points
    for i in range(len(m_final_final)):
        for j in range(i, len(m_final_final)):
            if m_final_final[i] - m_final_final[j] == 0:
                continue
            x_intersection = (c_final_final[i] - c_final_final[j]) / (m_final_final[j] - m_final_final[i])
            y_intersection = int(round(m_final_final[i] * x_intersection + c_final_final[i]))
            x_intersection = int(round(x_intersection))
            if x_intersection < 0 or x_intersection >= image.shape[1]:
                continue
            if y_intersection < 0 or y_intersection >= image.shape[0]:
                continue
            x_intersection_final_final.append(x_intersection)
            y_intersection_final_final.append(y_intersection)
    t_ = 3
    # drawing intersection points on the image
    for i in range(len(x_intersection_final_final)):
        d_ = t_
        x_intersection = x_intersection_final_final[i]
        y_intersection = y_intersection_final_final[i]
        while x_intersection + d_ < 0 or x_intersection + d_ > image.shape[1]:
            d_ = d_ - 1
        while y_intersection + d_ < 0 or y_intersection + d_ > image.shape[0]:
            d_ = d_ - 1
        image[y_intersection-d_:y_intersection+d_, x_intersection-d_:x_intersection+d_, 0] = 255
        image[y_intersection-d_:y_intersection+d_, x_intersection-d_:x_intersection+d_, 1] = 0
        image[y_intersection-d_:y_intersection+d_, x_intersection-d_:x_intersection+d_, 2] = 255
    return image, m_final_final, c_final_final


def run(image):
    """
    runs the whole program
    :param image: input image matrix
    :return: edges, acc_mat, lines_added, lines_added_final, dotted
    """
    edges = cv2.Canny(image, 300, 300)
    acc_mat, theta, rho = accumulator_matrix_creator(edges)
    m, c = lines_polar2mc(acc_mat, theta, rho)
    lines_added = line_drawer(image.copy(), m, c)
    dotted, m, c = find_intersections(image.copy(), m, c)
    lines_added_final = line_drawer(image.copy(), m, c)
    return edges, acc_mat, lines_added, lines_added_final, dotted


img1 = cv2.imread('./im01.jpg', cv2.IMREAD_UNCHANGED)
if img1 is None:
    raise Exception("Couldn't load the image")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

edges1, acc_mat1, lines_added1, lines_added_final1, dotted1 = run(img1)

img2 = cv2.imread('./im02.jpg', cv2.IMREAD_UNCHANGED)
if img2 is None:
    raise Exception("Couldn't load the image")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

edges2, acc_mat2, lines_added2, lines_added_final2, dotted2 = run(img2)

plt.imsave('res01.jpg', edges1, cmap='gray')
plt.imsave('res02.jpg', edges2, cmap='gray')

plt.imsave('res03-hough-space.jpg', acc_mat1, cmap='gray')
plt.imsave('res04-hough-space.jpg', acc_mat2, cmap='gray')

plt.imsave('res05-lines.jpg', lines_added1)
plt.imsave('res06-lines.jpg', lines_added2)

plt.imsave('res07-chess.jpg', lines_added_final1)
plt.imsave('res08-chess.jpg', lines_added_final2)

plt.imsave('res09-corners.jpg', dotted1)
plt.imsave('res10-corners.jpg', dotted2)
