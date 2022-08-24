import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt
from numpy import linalg

# for stroring the x values in various iteration in penalty function method.
x_series = []
# count for simultaneous linear dependancy hit in conjugate gradiant method.
simultaneousLDC = 0
# maximum limit for simultaneous linear dependancy hit in conjugate gradiant method.
simultaneousLDL = 5
# is linear dependancy check required (inside CG Method)?
isLDCRequired = True  # Default Value
# Stores Function Evaluation at all the iteration
pfmFCountSeries = []

np.seterr(all='raise')


class Objective:
    """Class which stores the objective of the problem
    """
    dictForObjectiveFunction = {}
    noOfFunctionEvaluations = 0
    penaltyFactor = 0
    noOfVarAllProblem = [2, 2, 8, 2, 2]

    def __init__(self, problemIndicator) -> None:
        self.problemIndicator = problemIndicator
        self.noOfVarCurProblem = self.noOfVarAllProblem[problemIndicator-1]

    def objectiveFunction(self, *x):
        """check whether values are already stored or not.

        Returns:
            number: objective function value
        """
        if x in self.dictForObjectiveFunction:
            return self.dictForObjectiveFunction.get(x)

        # when values are not stored calculate fresh.
        result = 0

        # Problem 1
        if self.problemIndicator == 1:
            result = (x[0]-10)**3+(x[1]-20)**3
        # Problem 2
        elif self.problemIndicator == 2:
            result = -(math.sin(2*math.pi*x[0])**3) * \
                math.sin(2*math.pi*x[1])/(x[0]**3*(x[0]+x[1]))
        # Problem 3
        elif self.problemIndicator == 3:
            result = x[0]+x[1]+x[2]
        # Problem 4 -> Test Case
        elif self.problemIndicator == 4:
            # test case
            result = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
        elif self.problemIndicator == 5:
            # test case f=x^2+y^2 without constraint it should be f* = 0
            result = x[0]**2+x[1]**2
        else:
            pass

        # store the new calculated value in the dictionary
        self.dictForObjectiveFunction[x] = result
        self.noOfFunctionEvaluations += 1
        return result

    def equilityConstraint(self, *x):
        """# Point to =0 constraint

        Returns:
            list: list of measures for the constraint functions.
        """
        return []

    def gtrInEquilityConstraint(self, *x):
        """# Point to >=0 constraint

        Returns:
            list: list of measures for the constraint functions.
        """
        constraintViolation = []
        if self.problemIndicator == 1:
            constraintViolation.extend([
                (x[0]-5)**2+(x[1]-5)**2-100,
                x[0]-13,
                x[1]-0
            ])
        elif self.problemIndicator == 2:
            constraintViolation.extend([
                x[0]-0,
                x[1]-0
            ])
        elif self.problemIndicator == 3:
            constraintViolation.extend([
                x[0]-0.01,
                x[1]-0.1,
                x[2]-0.1,
                x[3]-0.01,
                x[4]-0.01,
                x[5]-0.01,
                x[6]-0.01,
                x[7]-0.01
            ])
        elif self.problemIndicator == 4:
            constraintViolation.extend([
                (x[0]-5)**2+x[1]**2-26,
                x[0]-0,
                x[1]-0
            ])
        elif self.problemIndicator == 5:
            constraintViolation.extend([
                x[0]-2,
                x[1]-2
            ])
        else:
            pass
        return constraintViolation

    def lwrInEquilityConstraint(self, *x):
        """# Point to <=0 constraint

        Returns:
            list: list of measures for the constraint functions.
        """
        constraintViolation = []
        if self.problemIndicator == 1:
            constraintViolation.extend([
                (x[0]-6)**2+(x[1]-5)**2-82.81,
                x[0]-20,
                x[1]-4
            ])
        elif self.problemIndicator == 2:
            constraintViolation.extend([
                x[0]**2-x[1]+1,
                1-x[0]+(x[1]-4)**2,
                x[0]-10,
                x[1]-10
            ])
        elif self.problemIndicator == 3:
            constraintViolation.extend([
                -1+2.5*(x[3]+x[5]),
                -1+2.5*(-x[3]+x[4]+x[6]),
                -1+10*(-x[5]+x[7]),
                x[0]-10*x[0]*x[5]+8.3333252*x[3]-0.083333333,
                x[1]*x[3]-x[1]*x[6]-0.1250*x[3]+0.1250*x[4],
                x[2]*x[4]-x[2]*x[7]-0.25*x[4]+0.125,
                x[0]-1,
                x[1]-1,
                x[2]-1,
                x[3]-1,
                x[4]-1,
                x[5]-1,
                x[6]-1,
                x[7]-1
            ])

        else:
            pass
        return constraintViolation

    def constraintViolationStatus(self, *x):
        """# Provides status for constraint violation

        Returns:
            [[bool],[bool],[bool]]: equility, >=0 constraint, <=0 constraint
        """
        violation = []
        eqViolation = []
        gtrInEqViolation = []
        lwrInEqViolation = []
        for o in self.equilityConstraint(*x):
            if o != 0:
                eqViolation.append(True)
            else:
                eqViolation.append(False)
        for m in self.gtrInEquilityConstraint(*x):
            if m < 0:
                gtrInEqViolation.append(True)
            else:
                gtrInEqViolation.append(False)
        for n in self.lwrInEquilityConstraint(*x):
            if n > 0:
                lwrInEqViolation.append(True)
            else:
                lwrInEqViolation.append(False)

        violation.extend([eqViolation, gtrInEqViolation, lwrInEqViolation])
        return violation

    def isConstraintViolated(self, *x):
        """Gives the status whether any of the constraint is violated or not.

        Returns:
            bool: is constraint violated? yes -> True ; no -> False
        """
        for i in self.constraintViolationStatus(*x):
            for j in i:
                if j == True:
                    return True
                else:
                    continue
        return False

    def penaltyFunction(self, *x):
        """Penalty function using objective function, constraint violation and also penalty factor.


        Returns:
            number: function value.
        """
        penaltyTerm = 0
        for o in self.equilityConstraint(*x):
            pass
        for m in self.gtrInEquilityConstraint(*x):
            if m < 0:
                penaltyTerm = penaltyTerm + self.penaltyFactor*(m**2)
            else:
                pass
        for n in self.lwrInEquilityConstraint(*x):
            if n > 0:
                penaltyTerm = penaltyTerm + self.penaltyFactor*(n**2)
            else:
                pass

        return self.objectiveFunction(*x)+penaltyTerm


def partialDerivative(functionToOperate, variableIndicator, currentPoint):
    """
    This function will partially derive the a function
    with respect to variabel at a given point.
    It uses central difference method to implement the partial derivatives of first order.

    Args:
        functionToOperate (call back function): [function on which we will be differentiating]
        variableIndicator (int): [its an indicator for variable, starts from 1, with respect to which we will be partially differentiating]
        currentPoint (list): [current point at which we need to make the differentiation]

    Returns:
        [number]: [value]
    """
    deltaX = 10**-4
    pointOne = currentPoint.copy()
    pointTwo = currentPoint.copy()
    indicatorValue = currentPoint[variableIndicator-1]
    pointOne[variableIndicator-1] = indicatorValue + deltaX
    pointTwo[variableIndicator-1] = indicatorValue - deltaX
    return (functionToOperate(*pointOne)-functionToOperate(*pointTwo))/(2*deltaX)


def singleDerivative(functionToOperate, currentPoint):
    deltaX = 10**-4
    return (functionToOperate(currentPoint+deltaX)-functionToOperate(currentPoint-deltaX))/(2*deltaX)


def doubleDerivative(functionToOperate, currentPoint):
    deltaX = 10**-4
    return (
        singleDerivative(functionToOperate, currentPoint+deltaX) -
        singleDerivative(functionToOperate, currentPoint-deltaX))/(2*deltaX)


def gradiantOfFunction(functionToOperate, currentPoint):
    """Generate gradiant of a vector at a particular point

    Args:
        functionToOperate (call back function): function on which gradiant to be operate on.
        currentPoint (list): current point at which gradiant to be calculated.

    Returns:
        numpy array: gradiant vector
    """

    # Create a Zero matrix with no. of rows = no of variable in currentpoint
    # and only one column
    A = np.zeros((len(currentPoint), 1))
    for i in range(len(currentPoint)):
        A[i][0] = partialDerivative(functionToOperate, i+1, currentPoint)

    return A


def boundingPhaseMethod(functionToOperate, delta, a, b):
    """This is a Bracketing method.
    Which will be used to optimize a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        delta (float): Separation between two points
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        [a,b]: List containing the brackets [Lower,Upper].
    """

    deltaWithSign = None
    k = 0
    maxIteration = 10000
    initialRange = [a, b]
    maxAttmpt = 1000
    attmptCount = 0
    while True:
        # step 1
        if attmptCount >= maxAttmpt:
            # let region elimination method handle this task
            return [a, b]
        series = [0, 1, 2, 3, 4, 5]
        ind = math.trunc(random.uniform(0, len(series)))
        x_0 = series[ind]
        if (x_0 == a or x_0 == b):
            attmptCount = attmptCount + 1
            continue

        # step 2
        # In the below code there will be 3 function evaluations
        if functionToOperate(x_0 - abs(delta)) >= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) <= functionToOperate(x_0):
            deltaWithSign = + abs(delta)
        elif functionToOperate(x_0 - abs(delta)) <= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) >= functionToOperate(x_0):
            deltaWithSign = - abs(delta)
        else:
            attmptCount = attmptCount + 1
            continue

        while True:
            if k >= maxIteration:
                sys.exit("From Bounding Phase Method : No optimal point found")

            # step 3
            x_new = x_0 + 2**k*deltaWithSign
            if functionToOperate(x_new) < functionToOperate(x_0):
                k += 1
                x_0 = x_new
                continue
            else:
                # return in [x_lower,x_upper] format
                temp1 = x_new-(2**k)*1.5*deltaWithSign
                temp2 = x_new
                if temp1 > temp2:
                    return [temp2, temp1]
                else:
                    return [temp1, temp2]

    '''
    # The total no of function evaluation should be equal to (No. of iteration + 2)
    for this function as k starts with 0 so the total no of function evaluation = (k+1)+2
    '''


def intervalHalvingMethod(functionToOperate, epsinol, a, b):
    """This is a Region Elimination method.
    Which will be used to find the optimal solution for a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        epsinol (float): Very small value used to terminate the iteration
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        List: A list contains the bracket [lower,upper]
    """

    # step 1
    maxIteration = 10000
    x_m = (a+b)/2
    l = b-a
    no_of_iteration = 1
    initialRange = [a, b]
    while True:
        if no_of_iteration >= maxIteration:
            sys.exit("From Bounding Phase Method : No optimal point found")
        # step2
        x_1 = a+l/4
        x_2 = b-l/4
        while True:
            # step3
            if functionToOperate(x_1) < functionToOperate(x_m):
                b = x_m
                x_m = x_1
                break

            # step4
            if functionToOperate(x_2) < functionToOperate(x_m):
                a = x_m
                x_m = x_2
                break

            else:
                a = x_1
                b = x_2
                break

        # step5
        l = b-a
        if abs(l) < epsinol:
            return [a, b]
        else:
            no_of_iteration += 1
            continue


def newtonRaphsonMethod(functionToOperate, epsinon, initialPoint):
    # step 1
    #epsinol is set
    k = 1
    x_k = initialPoint

    while(True):
        # step 2 and step 3
        x_new = x_k - singleDerivative(functionToOperate,
                                       x_k) / doubleDerivative(functionToOperate, x_k)

        # step 5
        if abs(singleDerivative(functionToOperate, x_new)) <= epsinon or abs(x_k-x_new) <= 10**-3:
            return x_new

        x_k = x_new
        k = k+1


def changeToUniDirectionFunction(functionToOperate, x, s):
    """This is a function which will be used to scale the multivariable function
    into a single variable function using uni direction method.

    Args:
        functionToOperate (call back function): Multivariable objective function
        x (row vector): Position of initial point in [1,2,...] format
        s (row vector): Direction vector in [1,2,....] format

    Returns:
        function: Scaled functiion in terms of single variable -> a
    """
    return lambda a: functionToOperate(*(np.array(x)+np.multiply(s, a)))


def conjugateGradiantMethod(functionToOperate, limits, initialPoint):
    """This is an Gradiant Based Multi-Variable Optimisation Method.
    It is used to find the optimal solution of an objective function.

    Args:
        functionToOperate (call back function): Objective Function
        limits (list): in [lower,upper] format
        initialPoint (list): in [x0,x1,x2....] format
    """

    global simultaneousLDC, simultaneousLDL, isLDCRequired
    angleForDependencyInDegree = 1
    # step 1
    a, b = limits
    x_0 = list(initialPoint)
    epsinolOne = 10**-5
    epsinolTwo = 10**-5
    epsinolThree = 10**-5
    k = 0
    M = 10**4  # Maximum iteration for conjugate gradiant method.
    x_series = []  # store the x vextors
    x_series.append(x_0)

    # step2
    s_series = []  # store the direction vectors
    gradiantAtX_0 = gradiantOfFunction(functionToOperate, x_0)
    # Add unit vector along the gradient for new search direction
    s_series.append(np.divide(-gradiantAtX_0, np.linalg.norm(gradiantAtX_0)))
    # print(x_series[-1],gradiantAtX_0)
    # Extra termination condition *****
    if (np.linalg.norm(gradiantAtX_0)) <= epsinolThree:
        print(f"CG: Termination Point 1. Iterations Count -> {k}")
        return (x_0)

    # step3
    # convert multivariable function into single variable function
    s_0 = s_series[0][:, 0]
    newObjectiveFunction = changeToUniDirectionFunction(
        functionToOperate, x_0, s_0)

    # search for unidirection optimal point
    m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
    m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
    optimumPoint = (m+n)/2
    # print("Optimum Point",optimumPoint)

    x_1 = (np.array(x_0)+np.multiply(s_0, optimumPoint))
    # print(x_1)
    x_series.append(x_1)
    k = 1

    while(True):

        # step 4
        try:
            part_1_s = -gradiantOfFunction(functionToOperate, x_series[k])
            p = (np.linalg.norm(-part_1_s))**2
            q = gradiantOfFunction(functionToOperate, x_series[k-1])
            r = (np.linalg.norm(q))**2
            t = p/r
            part_2_s = np.multiply(s_series[k-1], t)
            s = part_1_s + part_2_s
            # s_series size will become to k+1
            s_series.append(np.divide(s, np.linalg.norm(s)))
        except:
            # if any kind of math error comes then return from here
            print("Math Error")
            return(x_series[k])

        # code to check linear independance
        if isLDCRequired:
            s_k = s_series[k][:, 0]  # row vector
            s_k_1 = s_series[k-1][:, 0]  # row vector

            dotProduct = s_k @ s_k_1
            factor = np.linalg.norm(s_k)*np.linalg.norm(s_k_1)
            if factor <= 10**-20:
                # in case factor = 0; to avoid math error.
                return x_series[k]
            finalDotProduct = dotProduct/factor
            finalDotProduct = round(finalDotProduct, 3)
            dependencyCheck = math.acos(
                finalDotProduct)*(180/math.pi)  # in degrees
            # print(dependencyCheck)
            if abs(dependencyCheck) < angleForDependencyInDegree:
                # Restart
                if simultaneousLDC > simultaneousLDL-1:
                    # if theere is simultaneous restart then stop executing any more
                    # and return what you have
                    print(
                        f"CG: LDC Overflowed. Will be procedding without LDC checking, onwards.")
                    isLDCRequired = False
                simultaneousLDC = simultaneousLDC + 1
                print(
                    f"CG: Linear Dependency Found! Restarting with {x_series[k]}")
                return conjugateGradiantMethod(functionToOperate, limits, x_series[k])

        # step 5
        # convert multivariable function into single variable function
        x_k = x_series[k]  # 1-D list
        s_k = s_series[k][:, 0]  # 1-D list
        newObjectiveFunction = changeToUniDirectionFunction(
            functionToOperate, x_k, s_k)

        # search for unidirection optimal point
        m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
        m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
        optimumPoint = (m+n)/2
        x_new = (np.array(x_k)+np.multiply(s_k, optimumPoint))
        x_series.append(x_new)  # x_series size will be k+2

        # step 6
        # check the terminate condition
        norm_1 = np.linalg.norm(np.array(x_series[k+1])-np.array(x_series[k]))
        norm_2 = np.linalg.norm(x_series[k])
        factor = np.linalg.norm(gradiantOfFunction(
            functionToOperate, x_series[k+1]))

        if norm_2 != 0:
            if norm_1/norm_2 <= epsinolTwo:
                print(f"CG: Termination Point 2. Iterations Count -> {k}")
                return x_series[k+1]

        if factor <= epsinolThree or k+1 >= M:
            print(f"CG: Termination Point 3. Iterations Count -> {k}")
            return x_series[k+1]
        else:
            k += 1
            continue

        break


def penaltyFunctionMethod(object, limits, initialPoint, epsinol_1=10 ^ -5, epsinol_2=10**-5, penaltyFactor=0.01, c=5):
    # step 1
    """Optimization method based on penalty function schema.
        which is if a set of constraint is violated we will add a penalty term to the function

    Args:
        object (call back function: pointer to a function): Penalty function
        limits (list): limits of variable, basically to be fend in to multi variable optimisation problem.
        initialPoint (list): initial point from which search to be started
        epsinol_1 (number): for termination condition 1
        epsinol_2 (number): for termination condition 2
        penaltyFactor (number): starting penalty factor
        c (number): incremental factor for penalty factor

    Returns:
        list: optimal point of the penalty function.

    """
    # step 1
    global x_series, simultaneousLDC, pfmFCountSeries
    # setting the initial value of penalty factor
    object.penaltyFactor = penaltyFactor
    k = 0  # iteration count
    x_series.append(initialPoint)
    maxIter = 50  # maximum iteration count after this it will find the any possible solution
    # after this no execution will be done, it will stop there.
    superMaxIter = 100
    # and it will retrun the same solution at iteration 0, at that point we will stop.

    while(True):
        fCountOffset = object.noOfFunctionEvaluations
        print(f"\nPFM:Penalty Factor -> {object.penaltyFactor}")
        print(f"PFM:Current Point -> {x_series[k]}\n")
        # step2
        # should be passed as an object->'object' to this method

        # step 3
        simultaneousLDC = 0
        isLDCRequired = True
        x_new = conjugateGradiantMethod(
            object.penaltyFunction, limits, x_series[k])
        x_series.append(x_new)
        pfmFCountSeries.append(object.noOfFunctionEvaluations-fCountOffset)

        # step 4
        term_1 = object.penaltyFunction(*x_series[k+1])
        if k != 0:
            # calculate the penalty with the previous value of penalty factor
            object.penaltyFactor = object.penaltyFactor/c
            term_2 = object.penaltyFunction(*x_series[k])
            # again reset the penalty factor as it is
            object.penaltyFactor = object.penaltyFactor*c
        else:
            # calculate the penalty with the previous value of penalty factor
            term_2 = object.penaltyFunction(*x_series[k])

        # termination conditions
        if abs(term_1-term_2) <= epsinol_1:
            print("PFM: Termination 1")
            return x_new

        if np.linalg.norm(np.array(x_series[k+1])-np.array(x_series[k]))/(np.linalg.norm(x_series[k])) <= epsinol_2:
            # If the two consecutive solutions are very close.
            # print(object.isConstraintViolated(*x_series[k+1]))
            if not object.isConstraintViolated(*x_series[k+1]):
                print("PFM: Termination 2")
                return x_new

        # step 5
        object.penaltyFactor = object.penaltyFactor*c
        k = k+1
        if k >= superMaxIter:
            sys.exit("\nPFM: SuperMax Iteration Overflowed.\n")
        if k >= maxIter:
            # if iteration count crosses the maximum limit, then retrun any possible value
            # from the point the current point is
            if object.isConstraintViolated(*x_new) != True:
                print(
                    "\nPFM: Iteration Count Overflowed! Returning closest possible value.\n")
                return x_new


def start():
    global x_axis
    problemSettings = {
        1: {"epsinol_1": 10**-4, "epsinol_2": 10**-3, "penaltyFactor": 100, "c": 10},
        2: {"epsinol_1": 10**-10, "epsinol_2": 10**-10, "penaltyFactor": 0.01, "c": 1.2},
        3: {"epsinol_1": 10**-5, "epsinol_2": 10**-3, "penaltyFactor": 18000, "c": 1.0001},
        4: {"epsinol_1": 10**-2, "epsinol_2": 10**-2, "penaltyFactor": 0.01, "c": 5},
        5: {"epsinol_1": 10**-5, "epsinol_2": 10**-5, "penaltyFactor": 0.01, "c": 5}
    }
    out = open(r"PFM_iterations.out", "w")
    out.write("x\tPenalty Function Value\tObjective Function Value\n")
    objectiveFunctionIndicator = int(input("Enter function indicator : \t"))
    limits = [-10, 10]
    if objectiveFunctionIndicator == 3:
        initialChoice = [1, 1, 1, 1, 1, 1, 1, 1]
    else:
        initialChoice = [1, 1]
    for i in range(1):
        p = Objective(objectiveFunctionIndicator)
        currentSettings = problemSettings.get(
            objectiveFunctionIndicator)  # a dictionary
        solution = penaltyFunctionMethod(
            p, limits, initialChoice, currentSettings['epsinol_1'], currentSettings['epsinol_2'], currentSettings['penaltyFactor'], currentSettings['c'])

        print(f"""
            Optimal Point             -> {solution}
            Greater Constraint (g>=0) -> {p.gtrInEquilityConstraint(*solution)}
            Lower Constraint  (h<=0)  -> {p.lwrInEquilityConstraint(*solution)} 
            Optimal Function Value    -> {p.objectiveFunction(*solution)}
            """)
        print("\t\t\t\t ---End---")

    # For ploting
    plt.subplot(2, 1, 1)
    x_axis = range(len(x_series))
    y1_axis = []
    y2_axis = []
    for x in x_series:
        y1_axis.append(p.penaltyFunction(*x))
        y2_axis.append(p.objectiveFunction(*x))
        out.write(f"{x}\t{p.penaltyFunction(*x)}\t{p.objectiveFunction(*x)}\n")
    plt.plot(x_axis, y1_axis, "r*-")
    plt.plot(x_axis, y2_axis, "b^-.")
    plt.xlabel("Iteration Count")
    plt.ylabel("Function Value")
    plt.legend(["Penalty Function", "Objective Function"])
    plt.subplot(2, 1, 2)
    plt.plot(pfmFCountSeries, "ys:")
    plt.xlabel("Iteration Count")
    plt.ylabel("Function Evaluations")
    plt.legend(["Function Evaluations"])
    plt.show()


start()
