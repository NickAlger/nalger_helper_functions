import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from nalger_helper_functions import AABBTree, plot_rectangle


K=2
num_boxes = 10000
num_pts = 4987

# Random boxes might be inside out. Doing this to check code can handle this without segfaulting
b_mins0 = np.array(np.random.randn(K, num_boxes), order='F')
b_maxes0 = np.array(np.random.randn(K, num_boxes), order='F')

AABB = AABBTree(b_mins0, b_maxes0)

q = np.random.randn(K)
inds = AABB.point_collisions(q)

qq = np.array(np.random.randn(K, num_boxes), order='F')
many_inds = AABB.point_collisions_vectorized(qq)

# Real boxes

K = 2
num_boxes = 191
num_pts = 549

bb0 = np.array(np.random.randn(K, num_boxes, 2), order='F')
b_mins = np.array(np.min(bb0, axis=2), order='F')
b_maxes = np.array(np.max(bb0, axis=2), order='F')

AABB = AABBTree(b_mins, b_maxes)
qq = 2.5 * np.array(np.random.randn(K, num_pts), order='F')
all_box_inds = AABB.point_collisions_vectorized(qq)

all_points_are_in_their_boxes = True
for kk in range(num_pts):
    qk = qq[:,kk]
    box_inds = all_box_inds[kk]
    qk_is_in_box = np.logical_and(np.all(b_mins[:, box_inds] <= qk[:, None]),
                                  np.all(qk[:, None] <= b_maxes[:, box_inds]))
    if not qk_is_in_box:
        all_points_are_in_their_boxes = False
        print('qk_is_in_box is false for kk=', kk)

print('all_points_are_in_their_boxes=', all_points_are_in_their_boxes)


bad_boxes_do_not_contain_points = True
for kk in range(num_pts):
    qk = qq[:,kk]
    box_inds = all_box_inds[kk]
    bad_box_inds = np.setdiff1d(np.arange(num_boxes, dtype=int), box_inds)
    qk_is_in_bad_box = np.any(np.logical_and(np.all(b_mins[:, bad_box_inds] <= qk[:, None], axis=0),
                                             np.all(qk[:, None] <= b_maxes[:, bad_box_inds], axis=0)))
    if qk_is_in_bad_box:
        bad_boxes_do_not_contain_points = False
        print('qk_is_in_bad_box for kk=', kk)

print('bad_boxes_do_not_contain_points=', bad_boxes_do_not_contain_points)


# More realistic setting, check correctness

num_boxes = int(1e3)
num_pts = int(1e4)

box_area = 1. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

qq = np.array(np.random.randn(K, num_pts), order='F')

AABB = AABBTree(b_mins, b_maxes)

all_box_inds = AABB.point_collisions_vectorized(qq)
all_points_are_in_their_boxes = True
for kk in range(num_pts):
    qk = qq[:,kk]
    box_inds = all_box_inds[kk]
    qk_is_in_box = np.logical_and(np.all(b_mins[:, box_inds] <= qk[:, None]),
                                  np.all(qk[:, None] <= b_maxes[:, box_inds]))
    if not qk_is_in_box:
        all_points_are_in_their_boxes = False
        print('qk_is_in_box is false for kk=', kk)

print('all_points_are_in_their_boxes=', all_points_are_in_their_boxes)


bad_boxes_do_not_contain_points = True
for kk in range(num_pts):
    qk = qq[:,kk]
    box_inds = all_box_inds[kk]
    bad_box_inds = np.setdiff1d(np.arange(num_boxes, dtype=int), box_inds)
    qk_is_in_bad_box = np.any(np.logical_and(np.all(b_mins[:, bad_box_inds] <= qk[:, None], axis=0),
                                             np.all(qk[:, None] <= b_maxes[:, bad_box_inds], axis=0)))
    if qk_is_in_bad_box:
        bad_boxes_do_not_contain_points = False
        print('qk_is_in_bad_box for kk=', kk)

print('bad_boxes_do_not_contain_points=', bad_boxes_do_not_contain_points)

# Check correctness for many tree sizes

print('Brute force correctness check')
num_pts = int(1e3)
all_points_are_in_their_boxes = True
bad_boxes_do_not_contain_points = True
for num_boxes in tqdm(range(1,70)):
    box_area = 1. / num_boxes
    box_h = np.power(box_area, 1./K)

    box_centers = np.random.randn(K, num_boxes)
    box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
    b_mins = np.array(box_centers - box_widths, order='F')
    b_maxes = np.array(box_centers + box_widths, order='F')

    qq = np.array(np.random.randn(K, num_pts), order='F')

    AABB = AABBTree(b_mins, b_maxes)

    all_box_inds = AABB.point_collisions_vectorized(qq)

    for kk in range(num_pts):
        qk = qq[:,kk]
        box_inds = all_box_inds[kk]
        qk_is_in_box = np.logical_and(np.all(b_mins[:, box_inds] <= qk[:, None]),
                                      np.all(qk[:, None] <= b_maxes[:, box_inds]))
        if not qk_is_in_box:
            all_points_are_in_their_boxes = False
            print('qk_is_in_box is false for kk=', kk, ', num_boxes=', num_boxes)

    bad_boxes_do_not_contain_points = True
    for kk in range(num_pts):
        qk = qq[:,kk]
        box_inds = all_box_inds[kk]
        bad_box_inds = np.setdiff1d(np.arange(num_boxes, dtype=int), box_inds)
        qk_is_in_bad_box = np.any(np.logical_and(np.all(b_mins[:, bad_box_inds] <= qk[:, None], axis=0),
                                                 np.all(qk[:, None] <= b_maxes[:, bad_box_inds], axis=0)))
        if qk_is_in_bad_box:
            bad_boxes_do_not_contain_points = False
            print('qk_is_in_bad_box for kk=', kk, ', num_boxes=', num_boxes)

print('all_points_are_in_their_boxes=', all_points_are_in_their_boxes)
print('bad_boxes_do_not_contain_points=', bad_boxes_do_not_contain_points)


# More realistic setting, timing

K=2
num_boxes = int(1e5)
num_pts = int(1e6)

print('num_boxes=', num_boxes, ', num_pts=', num_pts)

box_area = 30*(1. / num_boxes) # 1*(1. / num_boxes)
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

qq = np.array(np.random.randn(K, num_pts), order='F')

t = time()
AABB = AABBTree(b_mins, b_maxes)
dt_build = time() - t
print('dt_build=', dt_build)

t = time()
all_box_inds = AABB.point_collisions_vectorized(qq)
dt_point_collisions_vectorized = time() - t
print('dt_point_collisions_vectorized=', dt_point_collisions_vectorized)

# Recursive:
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.11991548538208008
# dt_first_point_intersection_vectorized= 5.488429069519043

# Iterative:
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.13932013511657715
# dt_first_point_intersection_vectorized= 5.261875629425049

# ALL collisions (not just first one)
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.1232149600982666
# dt_point_collisions_vectorized= 11.313974857330322

# ALL collisions (not just first one) AFTER GOING TO HEAP, no blocking yet
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.33385515213012695
# dt_point_collisions_vectorized= 13.094181537628174

# ALL collisions (not just first one) HEAP + BLOCKING
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.3034336566925049
# dt_point_collisions_vectorized= 17.75233292579651

# ALL collisions (not just first one) HEAP + NO BLOCKING + THREAD POOL
# box_area =  1*(1. / num_boxes)
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.3343532085418701
# dt_point_collisions_vectorized= 8.146560907363892

# Ball query

num_boxes = 351

box_area = 2. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

AABB = AABBTree(b_mins, b_maxes)

c = np.random.randn(K) / 4
r = 8.0 * box_h

intersections = AABB.ball_collisions(c, r)

plt.figure()
for k in range(num_boxes):
    plot_rectangle(b_mins[:, k], b_maxes[:, k], facecolor='b')

for jj in intersections:
    plot_rectangle(b_mins[:, jj], b_maxes[:, jj], facecolor='r')

circle1 = plt.Circle(c, r, edgecolor='k', fill=False)
plt.gca().add_patch(circle1)

big_box_min = np.min(b_mins, axis=1)
big_box_max = np.max(b_maxes, axis=1)
plt.xlim(big_box_min[0], big_box_max[0])
plt.ylim(big_box_min[1], big_box_max[1])
plt.show()


# Ball query timing

K = 2

num_boxes = int(1e6)
num_balls = int(1e6)

box_area = 2. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

AABB = AABBTree(b_mins, b_maxes)

ball_centers = np.array(np.random.randn(K, num_balls), order='F')
ball_radii = np.array(6.0 * box_h * np.random.randn(num_balls), order='F')

t = time()
all_collisions = AABB.ball_collisions_vectorized(ball_centers, ball_radii)
dt_ball = time() - t
print('num_boxes=', num_boxes, ', num_balls=', num_balls, ', dt_ball=', dt_ball)

mean_num_collisions_ball = np.mean([len(c) for c in all_collisions])
print('mean_num_collisions_ball=', mean_num_collisions_ball)

# np.mean([len(x) for x in all_collisions])

# AFTER adding thread pooling 11/16/21
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 1.9799399375915527

# AFTER removing templated dimension, plus using FIFO queue instead of vector 11/12/21
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 6.3552868366241455

# AFTER removing templated dimension 11/12/21
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 7.004610538482666

# BEFORE removing templated dimension 11/12/21
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 4.182887315750122

# std::vector<int> for intersections, radius
# num_boxes= 1000000 , num_balls= 100000 , dt_ball= 0.7881059646606445

# std::vector<int> for intersections, radius squared
# num_boxes= 1000000 , num_balls= 100000 , dt_ball= 0.7624087333679199

# VectorXi for intersections, radius squared
# num_boxes= 1000000 , num_balls= 100000 , dt_ball= 0.4457828998565674


# BEFORE REMOVING nodes_under_consideration:
# all_good_points_are_in_their_boxes= True
# bad_points_are_the_points_outside_all_boxes= True
# all_good_points_are_in_their_boxes= True
# bad_points_are_the_points_outside_all_boxes= True
# num_boxes= 100000 , num_pts= 1000000
# dt_build= 0.12497305870056152
# dt_first_point_intersection_vectorized= 0.5357162952423096
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 3.665412664413452

# AFTER:
# all_good_points_are_in_their_boxes= True
# bad_points_are_the_points_outside_all_boxes= True
# all_good_points_are_in_their_boxes= True
# bad_points_are_the_points_outside_all_boxes= True
# num_boxes= 100000 , num_pts= 1000000
# dt_build= 0.12599444389343262
# dt_first_point_intersection_vectorized= 0.5189225673675537
# num_boxes= 100000 , num_balls= 1000000 , dt_ball= 3.6354219913482666

# Box query

num_boxes = 351

box_area = 2. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

AABB = AABBTree(b_mins, b_maxes)

qc = np.random.randn(K) / 4
qmin = qc - np.random.rand(K)
qmax = qc + np.random.rand(K)

intersections = AABB.box_collisions(qmin, qmax)

plt.figure()
for k in range(num_boxes):
    plot_rectangle(b_mins[:, k], b_maxes[:, k], facecolor='b')

for jj in intersections:
    plot_rectangle(b_mins[:, jj], b_maxes[:, jj], facecolor='r')

plot_rectangle(qmin, qmax)

big_box_min = np.min(b_mins, axis=1)
big_box_max = np.max(b_maxes, axis=1)
plt.xlim(big_box_min[0], big_box_max[0])
plt.ylim(big_box_min[1], big_box_max[1])
plt.show()


# Box query timing

K = 2

num_boxes = int(1e6)
num_query_boxes = int(1e6)

box_area = 2. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(K, num_boxes)
box_widths = box_h * np.abs(np.random.randn(K, num_boxes))
b_mins = np.array(box_centers - box_widths, order='F')
b_maxes = np.array(box_centers + box_widths, order='F')

AABB = AABBTree(b_mins, b_maxes)

query_centers = np.random.randn(K, num_query_boxes) / 4
query_mins = np.array(query_centers - 8.0 * box_h * np.random.rand(K, num_query_boxes), order='F')
query_maxes = np.array(query_centers + 8.0 * box_h * np.random.rand(K, num_query_boxes), order='F')

t = time()
all_collisions = AABB.box_collisions_vectorized(query_mins, query_maxes)
dt_box = time() - t
print('num_boxes=', num_boxes, ', num_query_boxes=', num_query_boxes, ', dt_box=', dt_box)

mean_num_collisions_box = np.mean([len(c) for c in all_collisions])
print('mean_num_collisions_box=', mean_num_collisions_box)