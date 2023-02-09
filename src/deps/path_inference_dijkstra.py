import numpy as np
import os
from scipy.interpolate import interpn
import random
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from collections import Counter
from .dijkstra_modified import Graph
# from dijkstra_modified import Graph
import time
# sparseimport
from Qt import QtCore, QtWidgets


class Path(QtCore.QObject):
    finished = QtCore.Signal()
    exit = QtCore.Signal()
    progress = QtCore.Signal(float, float, bool, bool, bool)
    status = QtCore.Signal(bool)
    def __init__(self, input_data, learn, wiener_width, median_width, resolution, spawn, num_minima, coupling_const, spring_const):
        self.dataPth = input_data
        self.data = input_data
        self.resolution = resolution
        self.neighbours = 10
        self.pad_factor = 0.1
        self.median_width = median_width
        self.wiener_width = wiener_width
        self.truncation_width = 9
        self.learn_rate = learn
        self.spawn = spawn
        self.beads = 40
        self.coupling_const = coupling_const
        self.spring_const = spring_const
        self.num_minima = num_minima
        print("Initialised with: learn %s, wiener %s, median %s, res %s, spawn %s, minima %s, coupling %s, spring %s" %
              (learn,wiener_width,median_width,resolution,spawn,num_minima,coupling_const,spring_const))
        np.seterr(divide='ignore', invalid='ignore')
        super().__init__()

    def load_data(self, path):
        if os.path.isfile(path):
            d = np.load(path, allow_pickle=True)
            # d = np.genfromtxt(path, delimiter=',')
            return d
        else:
            return np.random.randint(50, size=(64, 64))

    # def NN_approach(self):
    #     random.seed(os.urandom(128))
    #
    #     def data_coord2view_coord(p, resolution, pmin, pmax):
    #         dp = pmax - pmin
    #         dv = (p - pmin) / dp * resolution
    #         return dv
    #
    #     self.data = self.load_data(self.dataPth)
    #
    #     resolution = 250
    #     xs = self.data[:, 0]
    #     ys = self.data[:, 1]
    #     extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
    #
    #     xv = data_coord2view_coord(xs, resolution, extent[0], extent[1])
    #     yv = data_coord2view_coord(ys, resolution, extent[2], extent[3])
    #
    #     search = self.kNN2DDens(xv, yv, resolution, 128)*1000
    #     minDim, maxDim = [0, search.shape[1]]
    #
    #     # search = global_penalty - penalty
    #     # search = 30 * self.np_norm(search)
    #     self.show(search, 'search')
    #     plt.show()
    #
    #     self.gradient(search)
    #
    #     # self.show((self.gradList[0]), 'gradient X')
    #     # self.show((self.gradList[1]), 'gradient Y')
    #
    #     # grad = (self.gradList[0]+self.gradList[1])/2
    #     # self.show(grad, 'gradient')
    #
    #     # fig, ax = self.show(self.gradList[0], 'grad[0]')
    #     # fig2, ax2 = self.show(self.gradList[1], 'grad[1]')
    #     fig, ax = self.show(search, 'search')
    #     # self.scatter(self.data, 'trajectory')
    #     # self.show(histo, 'histo_inverse')
    #
    #     step = 5
    #     buffer = 0.01
    #     lowBound = int(minDim + buffer * maxDim)
    #     highBound = int(maxDim - buffer * maxDim)
    #     for trial in range(50):
    #         random.seed()
    #         # r = random.random()
    #         # b = random.random()
    #         # g = random.random()
    #         # color = (r, g, b)
    #         point = (random.randint(lowBound, highBound), random.randint(lowBound, highBound))
    #         # point = (lowBound+random.random()*(highBound-lowBound)), (lowBound+random.random()*(highBound-lowBound))
    #
    #         vector, self.pos = self.gradient_descent(point, 0.05)
    #         # if (np.abs(np.linalg.norm(self.pos[0]-self.pos[-1])) <= 0.5):
    #         # continue
    #         # else:
    #         colors = cm.gnuplot2(np.linspace(0, 1, len(self.pos[0::step, 0])))
    #         ax.scatter(self.pos[0::step, 0], self.pos[0::step, 1], zorder=2, color=colors, s=4)
    #         ax.plot(self.pos[0::step, 0], self.pos[0::step, 1], zorder=1, color=(0, 0, 0))
    #
    #         # ax2.scatter(self.pos[0::step, 0], self.pos[0::step, 1], zorder=2, color=colors, s=4)
    #         # ax2.plot(self.pos[0::step, 0], self.pos[0::step, 1], zorder=1, color=(0, 0, 0))
    #         # ax3.scatter(self.pos[0::step, 0], self.pos[0::step, 1], zorder=2, color=colors, s=4)
    #         # ax3.plot(self.pos[0::step, 0], self.pos[0::step, 1], zorder=1, color=(0, 0, 0))
    #         if trial % 25 == 0:
    #             print("Stepping by %s" % trial)
    #     plt.show()

    # def histogram(self, input, pixel):
    #     # x_min, y_min = np.amin(input, axis=0)
    #     # x_max, y_max = np.amax(input, axis=0)
    #     # pltRange = (x_max - x_min), (y_max - y_min)
    #     # bins = int(pltRange[0] / pixel )
    #     small = np.array(input[:,:4])
    #     return np.histogramdd(small, bins=self.bins)

    # def run_embedding(self, input):
    #     import umap
    #     operator = umap.UMAP(random_state=42, verbose=1)
    #     return operator.fit_transform(input)

    # def show(self, input, name):
    #     plt.subplots(figsize=(8, 7))
    #     plt.pcolormesh(input, shading='auto', cmap=plt.cm.PRGn)
    #     plt.savefig('./temp/' + str(name) + '.png')
    #     # return fig, ax

    def scatter(self, data, name):
        import matplotlib.pyplot as plt
        plt.scatter(data[:,0], data[:,1])
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def norm(self, arr):
        scaler = StandardScaler()
        scaler.fit(arr)
        return scaler.transform(arr)

    def np_norm(self,arr):
        n = np.linalg.norm(arr)
        arr = arr/n
        min = np.amin(arr)
        arr = arr - min
        max = np.amax(arr)
        arr = arr/max
        return arr

    def boundary_conditions(self, vector, boundary):
        '''
        Checks vector is within boundary condition
            returns the same vector if within boundary
            returns a new vector mapped to the periodically defined position
        '''
        x_max = np.amax(boundary[0], axis=0)
        x_min = np.amin(boundary[0], axis=0)
        y_max = np.amax(boundary[1], axis=0)
        y_min = np.amin(boundary[1], axis=0)
        v0 = vector[0]
        v1 = vector[1]

        if (v0 < x_max) and (v0 > x_min) and (v1 < y_max) and (v1 < y_max):
            # print("Vector OK: %s" % str(vector))
            return vector
        else:
            # print("Vector NOT OK: %s" % str(vector))
            if (v0 < x_min):
                v0 = v0 + x_max
            if (v0 > x_max):
                v0 = v0 - x_max
            if (v1 < y_min):
                v1 = v1 + y_max
            if (v1 > y_max):
                v1 = v1 - y_max
            vector = (v0, v1)
            # print("Vector now: %s" % str(vector))
            return vector

    def gradient(self, input):
        self.gradList = np.gradient(input)
        self.points = []
        for grad in self.gradList: #Change to np.mgrid
            pts = [np.arange(0, grad.shape[axis], 1) for axis in range(len(grad.shape))]
            self.points.append(pts)

    def gradient_increment(self, vector):
        diffVec = []
        for i, grad in enumerate(reversed(self.gradList)):
            vector = self.boundary_conditions(vector, self.points[i])
            partialDiff = interpn(self.points[i], grad.T, vector)[0]
            diffVec.append(partialDiff)
        diffVec = np.array(diffVec)

        return vector, diffVec

    def gradient_descent(self, start, learn_rate, n_iter=100, tolerance=1e-3):
        vector = start
        pos = start
        strike = 0
        for _ in range(n_iter):
            vector, diffVec = self.gradient_increment(vector)

            diff = -learn_rate * diffVec

            if np.all(np.abs(diff) <= tolerance):
                strike += 1
                #xRand = self.pixel*(random.random()*2 - 1)
                #yRand = self.pixel*(random.random()*2 - 1)
                #vector = (vector[0]+xRand, vector[1]+yRand)
                if strike == 10:
                    break

            vector += diff
            pos = np.vstack([pos, vector])
            # if _ % 10 == 0:
            #     print("Stepping by %s" % diff)
            QtCore.QCoreApplication.processEvents()
        return vector, pos

    def cost_function(self, histo):
        C = np.ones_like(histo)
        costFunc = []
        for x_row, row in enumerate(C):
            for x_ele, ele in enumerate(row):
                ref = [x_row, x_ele]
                distances = np.fromiter(
                    (np.linalg.norm(np.array(ref) - np.array([i, j]))
                     for i, row in enumerate(C) for j, item in enumerate(row)), float)
                distances[distances == 0] = 1
                distances = distances ** -2
                distanceMatrix = distances.reshape(C.shape)

                weightMatrix = np.multiply(histo, distanceMatrix)
                SUM = np.sum(np.sum(weightMatrix, axis=0), axis=0)
                costFunc.append(SUM)

        costFunc = np.array(costFunc).reshape(C.shape)
        return costFunc

    def cost_function_local_trunc(self, histo, window):
        wX, wY = window, window #3,5,7,9,15,25
        pad = int((window-1)/2)
        A = np.ones_like(histo)
        C = np.pad(A, pad, mode='constant', constant_values=0)
        costFunc = []
        histoPad = np.pad(histo, pad, mode='constant', constant_values=0)
        for x_row, row in enumerate(C, start=pad):
            if x_row >= C.shape[0]-pad:
                break
            for x_ele, ele in enumerate(row, start=pad):
                if x_ele >= C.shape[1]-pad:
                    break
                ref = [x_row, x_ele]
                distances = np.fromiter(
                    (np.linalg.norm(np.array(ref) - np.array([x_row-pad+i, x_ele-pad+j]))
                     for i in range(wX) for j in range(wY)), float)
                distances[distances == 0] = 1
                distances = distances ** -2
                distanceMatrix = distances.reshape(wX,wY)
                histoSlice = histoPad[x_row-pad:x_row+pad+1, x_ele-pad:x_ele+pad+1]

                weightMatrix = np.multiply(histoSlice, distanceMatrix)
                SUM = np.sum(np.sum(weightMatrix, axis=0), axis=0)
                costFunc.append(SUM)

        costFunc = np.array(costFunc).reshape(A.shape)
        # norm = np.linalg.norm(costFunc)
        # costFunc = costFunc / norm
        return costFunc

    def n_dim_convolution(self, histo, window):
        from scipy import signal
        n = len(histo.shape)
        w = (window-1)/2
        grid = np.mgrid[tuple(slice(1 - w, w, 1) for _ in range(n))]
        grid_sq = grid ** 2
        sum = grid_sq[0]
        for axis, _ in enumerate(grid_sq):
            if axis == len(grid_sq)-1:
                break
            sum = np.add(sum, grid_sq[axis+1])
        sum[sum == 0] = 1
        kernel = 1/sum
        return signal.fftconvolve(histo, kernel, mode='same')

    def kNN2DDens(self, input, resolution, neighbours, dim=2):
        # Create the tree
        tree = cKDTree(input.T)
        # Find the closest nnmax-1 neighbors (first entry is the point itself)
        grid = np.mgrid[tuple(slice(0, resolution, 1) for _ in range(dim))].T.reshape(resolution ** dim, dim)
        dists = tree.query(grid, neighbours)

        # Inverse of the sum of distances to each grid point.
        inv_sum_dists = 1. / (dists[0]**2).sum(1)

        # Reshape
        im = inv_sum_dists.reshape(tuple(resolution for _ in range(dim)))
        return np.pad(im, int(self.pad_factor * self.resolution), mode='constant', constant_values=0)

    def analyse_search_paths(self, list_of_paths):
        '''
        Discard searches which got stuck
        '''
        tolerance = 0.03*self.resolution
        n_last = 10
        end_points = []
        for traj in list_of_paths:
            if (np.linalg.norm(traj[0] - traj[-1]) >= tolerance) and (np.linalg.norm(traj[-n_last] - traj[-1]) < tolerance):
                rounded = np.around(traj[-1], decimals=0)
                end_points.append((rounded[0], rounded[1]))
        return end_points

    def nudge_elastic_band(self, minimum_A, minimum_B, n_iter, learn1, learn2, N_beads, spring_const):
        #Interpolate Nbeads points along a trajectory in n-space between minimum_A and minimum_B
        norm = np.linalg.norm(minimum_B-minimum_A)
        unit_vector = (minimum_B - minimum_A) / norm
        start = minimum_A
        beads = np.zeros_like(start)
        for b in range(N_beads+1):
            bead = start + b*unit_vector*(norm/N_beads)
            beads = np.vstack((beads, bead))
        beads = beads[1:]

        # Spring force between beads so they don't fly away
        def spring_force(prev_bead, curr_bead, next_bead, spring_k):
            '''
                prev_bead position of previous bead
                mid_bead position of bead
                next_bead position of next bead
                k = spring constant
            '''
            #Determine distance between beads
            r1 = np.linalg.norm(next_bead-curr_bead)
            # Define orthogonal basis, (i,j,k) where k is parallel to string vector (i.e. unit vector) between beads.
            k1_hat = (next_bead - curr_bead) / r1
            i_hat = np.random.randn(len(k1_hat))
            i_hat -= i_hat.dot(k1_hat) * k1_hat
            i_hat /= np.linalg.norm(i_hat)
            j_hat = np.cross(k1_hat, i_hat)
            # force in the direction of xn-xi i.e. parallel with the bead string
            pos_force = -2 * spring_k * r1 * k1_hat

            #_______________________________________________________#
            # Determine distance between beads
            r2 = np.linalg.norm(prev_bead - curr_bead)
            # Define orthogonal basis, (i,j,k) where k is parallel to string vector (i.e. unit vector) between beads.
            k2_hat = (prev_bead - curr_bead) / r2
            # force in the direction of xn-xi i.e. parallel with the bead string
            neg_force = -2 * spring_k * r2 * k2_hat

            # _______________________________________________________#
            net_force = pos_force + neg_force
            return net_force, i_hat, j_hat, k1_hat

        def projection(u, v):
            #Project u onto v
            proj = (np.dot(u, v) / np.linalg.norm(v) ** 2) * v
            return proj

        def energy_gradient_increment(vector, gradList, points):
            diffVec = []
            for i, grad in enumerate(reversed(gradList)):
                vector = self.boundary_conditions(vector, points[i])
                partialDiff = interpn(points[i], grad.T, vector)[0]
                diffVec.append(partialDiff)
            diffVec = np.array(diffVec)
            return vector, diffVec

        def string_gradient_descent(beads_array, learn_rate, learn_rate2, n_iter):
            beads = np.copy(beads_array)

            for _ in range(n_iter):
                new_beads = np.copy(beads[0])
                total_diff = 0
                for i,bead in enumerate(beads[1:-1]):
                    S_GradVector, i_hat, j_hat, k_hat = spring_force(beads[i], beads[i + 1], beads[i + 2], spring_const)
                    #Now calculate energy force on bead due to gradient
                    bead, E_GradVector = energy_gradient_increment(bead, self.gradList, self.points)
                    E_GradVectorProjected = E_GradVector - projection(E_GradVector, k_hat)
                    diff = (-1*learn_rate * S_GradVector) + (-1*learn_rate2 * E_GradVectorProjected)
                    new_bead = bead + diff
                    new_beads = np.vstack((new_beads, new_bead))
                    total_diff =+ np.abs(diff)
                    QtCore.QCoreApplication.processEvents()
                if np.linalg.norm(total_diff) < 0.00025*N_beads:
                    break
                new_beads = np.vstack((new_beads, beads[-1]))
                beads = new_beads
            return beads

        minimised_beads = string_gradient_descent(beads, learn1, learn2, n_iter)
        return beads, minimised_beads

    def energy_integral(self, path, landscape):
        total_energy = 0
        for point in path:
            energy_value = np.linalg.norm(interpn(self.points[0], landscape.T, point)[0])
            for i, grad in enumerate(reversed(self.gradList)):
                energy_gradient = np.linalg.norm(interpn(self.points[i], grad.T, point)[0])
                energy_increment = energy_gradient + energy_value
                total_energy += energy_increment
        return total_energy

    def vector_energy_integral(self, path, landscape):
        total_energy = 0
        for i,point in enumerate(path[:-1]):
            energy_value = np.linalg.norm(interpn(self.points[0], landscape.T, point)[0])
            r = (path[i+1] - path[i])
            del_energy = []
            for axis, grad in enumerate(reversed(self.gradList)):
                partial_diff_energy = np.linalg.norm(interpn(self.points[axis], grad.T, path[i])[0])
                del_energy.append(partial_diff_energy)

            del_energy = np.array(del_energy)
            energy_increment = np.dot(r, del_energy)+energy_value
            total_energy += energy_increment

        return total_energy

    def split_by_stopovers(self, path, minima):
        route = []
        tolerance = 0.01*self.resolution
        for minimum in minima:
            for position, coords in enumerate(path):
                if (np.abs(np.linalg.norm(coords - minimum)) <= tolerance):
                    route.append(np.array([position, minimum[0], minimum[1]]))
                    break
        route = np.array(route)
        sorted_route = route[np.argsort(route[:, 0])]
        unique_sorted_route = sorted_route[np.unique(sorted_route[:, 0], axis=0, return_index=True)[1]]

        if unique_sorted_route.shape[0] >= 2:
            return unique_sorted_route
        else:
            return np.array([])

    def points2indices(self, list_of_paths, final_space):
        '''
        Get nearest neighbour to bead points/coordinates
        :return: List of lists. Each list consists of a trajectory (indices of original data)
        '''
        traj_list = []
        tree = cKDTree(self.data)
        for path in list_of_paths:
            sub_path = []
            for coord in path:
                coord = self.map2original(coord, self.data, final_space)
                dist, ind = tree.query(coord, k=1)
                sub_path.append(ind)
            traj_list.append(sub_path)
        return traj_list

    def preprocessing(self, data):
        pass

    def resample(self, data):
        xs = data[:, 0]
        ys = data[:, 1]
        xmin = np.min(xs)
        xmax = np.max(xs)
        ymin = np.min(ys)
        ymax = np.max(ys)

        dy = ymax - ymin
        y = (ys - ymin) / dy * self.resolution

        dx = xmax - xmin
        x = (xs - xmin) / dx * self.resolution
        return np.array([x, y])

    def map2original(self, coord, data, final_space):
        cx, cy = coord
        xs = data[:, 0]
        ys = data[:, 1]
        xmin = np.min(xs)
        xmax = np.max(xs)
        ymin = np.min(ys)
        ymax = np.max(ys)

        dy = ymax - ymin
        dx = xmax - xmin

        pad_factor = 0.5*(final_space.shape[0] - self.resolution)

        y = ((cy - pad_factor) * dy / self.resolution) + ymin
        x = ((cx - pad_factor) * dx / self.resolution) + xmin
        return np.array([x,y])

    def costFuncApproach(self):
        self.status.emit(False)
        t0 = time.time()
        self.progress.emit(120, t0, True, True, False)
        random.seed(os.urandom(128))
        print("Converting observations into a pseudoEnergy landscape...")
        # self.data = self.load_data(self.dataPth)

        array = self.resample(self.data)

        histo = self.kNN2DDens(array, self.resolution, self.neighbours)
        # self.show(histo, 'A_no_data2view')


        histo = self.np_norm(histo)
        # self.show(histo, 'B_normalised')


        from scipy import signal
        histo = signal.wiener(histo.astype('float64'), self.wiener_width)
        # self.show(histo, 'C_modified')


        histo = np.pad(histo, int(self.pad_factor*self.resolution), mode='constant', constant_values=0)
        minDim, maxDim = [0, histo.shape[1]]

        #First convolution
        penalty = signal.medfilt(histo, self.median_width)

        # self.show(penalty, 'D_median filter (7)')


        # Use kernel density estimation method to convert frequency distribution to penalty function "energy lanscape"
        #penalty = self.cost_function_local_trunc(histo, 9)
        penalty = self.n_dim_convolution(penalty, self.truncation_width)

        # self.show(penalty, 'E_convolution with inverse distance')


        penalty = self.np_norm(penalty)

        grid_x, grid_y = np.mgrid[-10:10:maxDim*1j, -10:10:maxDim*1j]
        def func(x,y):
            a = 15
            b = 15
            z = ((1/a)*(x**2)+(1/b)*(y**2))
            return z
        global_penalty = func(grid_x, grid_y)
        global_penalty = 0.1*self.np_norm(global_penalty)

        search = global_penalty-penalty
        search = 30*self.np_norm(search)

        self.gradient(search)
        print("Searching pseudoEnergy landscape...")

        # self.show(search, 'F_search')

        buffer = 0.01
        lowBound = int(minDim + buffer * maxDim)
        highBound = int(maxDim - buffer * maxDim)
        list_of_paths = []
        print("Searching all local minima (metastable or stable conformations)")
        for trial in range(self.spawn):
            random.seed()
            point = (random.randint(lowBound, highBound), random.randint(lowBound, highBound))
            vector, self.pos = self.gradient_descent(point, self.learn_rate)

            if trial % 25 == 0:
                print("Step %s. Total complete: %s %%" % (trial, round(100*trial/self.spawn, 1)), end="\r")
            list_of_paths.append(self.pos)

        print("\nMinima found")

        self.end_point_frequency = self.analyse_search_paths(list_of_paths)
        minima = Counter(self.end_point_frequency)
        top_hits = dict(sorted(minima.items(), key=lambda item: item[1], reverse=True))
        conform_minima = []
        for key in list(top_hits)[:self.num_minima]:
            #print("key: %s , value: %s" % (key, minima[key]))
            conform_minima.append(key)
        conform_minima = np.array(conform_minima)


        trial = 0
        look_up_table = []
        print("Determine the least energy path between permutations of local minima")
        for n,i in enumerate(conform_minima):
            for m,j in enumerate(conform_minima):
                trial += 1
                if not (i == j).all():
                    path, new_path = self.nudge_elastic_band(i, j, int(self.beads*1.2), self.learn_rate, self.coupling_const, self.beads, self.spring_const)
                    stopover_list = self.split_by_stopovers(new_path, conform_minima)

                    if stopover_list.size:
                        for stopover,coords in enumerate(stopover_list[:-1]):
                            start_of_path = int(stopover_list[stopover][0])
                            stop_of_path = int(stopover_list[stopover + 1][0])

                            minima_at_start = np.array([stopover_list[stopover][1], stopover_list[stopover][2]])
                            minima_at_stop = np.array([stopover_list[stopover+1][1], stopover_list[stopover+1][2]])

                            sub_path = new_path[start_of_path:stop_of_path+1]
                            total = np.around(self.vector_energy_integral(sub_path, search), decimals=1)

                            look_up_table.append((minima_at_start[0], minima_at_start[1],
                                                  minima_at_stop[0], minima_at_stop[1],
                                                  total, sub_path))

                    else:
                        total = np.around(self.vector_energy_integral(new_path, search), decimals=1)

                        look_up_table.append((i[0], i[1],
                                              j[0], j[1],
                                              total, new_path))

                if trial % 10 == 0:
                    print("Minimising beads: Step %s. Total complete: %s %%" %
                          (trial, round(100 * trial / len(conform_minima)**2, 1)), end = "\r")

        look_up_table = np.array(look_up_table, dtype=object)
        print("Paths of least energy identified")

        # Sort the look up table from lowest energy path to highest energy path
        energy_ascending_look_up_table = look_up_table[np.argsort(look_up_table[:, 4])]
        index = np.array(energy_ascending_look_up_table[:, :4], dtype=np.float64)
        # Remove redundant paths by taking subpath with least energy
        look_up_table = energy_ascending_look_up_table[np.unique(index, axis=0, return_index=True)[1]]

        # Unique list
        A = np.array([np.array([ele[0], ele[1]]) for ele in look_up_table])
        B = np.array([np.array([ele[2], ele[3]]) for ele in look_up_table])
        merge = np.vstack((A, B))
        non_redundant_list = np.unique(merge, axis=0)

        # Convert look_up_table to nodes and edges for dijkstra's algorithm
        node_table = []
        for row in look_up_table:
            for node, point in enumerate(non_redundant_list):
                if point[0] == row[0] and point[1] == row[1]:
                    n = node
            for node, point in enumerate(non_redundant_list):
                if point[0] == row[2] and point[1] == row[3]:
                    m = node
            node_table.append([n, m, row[4], row[5]])
        node_table = np.array(node_table, dtype=object)

        # Initialise the graph and weights for dijkstra's algorithm
        graph = Graph(len(non_redundant_list))
        for edge in node_table:
            graph.addEdge(edge[0], edge[1], edge[2])


        l = np.arange(self.num_minima)
        np.random.shuffle(l)
        final_paths = []
        for i in l:
            # Plot the paths
            graph.dijkstra(i)
            minimal_tree = graph.tree
            np.random.shuffle(minimal_tree)
            r = random.random()
            b = random.random()
            g = random.random()
            colour = (r, g, b)

            count = 0
            for branch in minimal_tree:
                dijkstras_route = [0, 0]

                if len(branch[2]) > 2:
                    for step, _ in enumerate(branch[2][:-1]):
                        nodeA = branch[2][step]
                        nodeB = branch[2][step+1]
                        sub_path = node_table[np.where((node_table[:, 0] == nodeA) * (node_table[:, 1] ==  nodeB))][0][3]
                        dijkstras_route = np.vstack((dijkstras_route, sub_path))
                    # plt.scatter(dijkstras_route[1:, 0], dijkstras_route[1:, 1], s=10, zorder=3, color=colour)
                    # plt.plot(dijkstras_route[1:, 0], dijkstras_route[1:, 1], zorder=2, color=colour)
                    final_paths.append(dijkstras_route[1:])

        # plt.savefig('./temp/G_last.png')
        self.trajectories = self.points2indices(final_paths, search)

        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)


if __name__ == '__main__':
    print("Running...")
    p = Path('./z_manifold.large.pkl')
    p.costFuncApproach()

