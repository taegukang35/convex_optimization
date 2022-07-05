import numpy as np
import matplotlib.pyplot as plt
import random

# SVM에 사용할 데이터 만들기
# 50개씩 (8,0), (0,8)에 몰리도록 랜덤하게 생성
# (0,8)에 가까운 점은 S_i = 1, (8,0)에 가까운 점은 S_i = -1
n = 100
r1,r2 = np.random.randn(n//2,1),np.random.randn(n//2,1)+8
s1,s2 = np.ones((n//2,1)),-np.ones((n//2,1))
r = np.hstack((np.vstack((r1,r2)),np.vstack((r2,r1))))
S = np.vstack((s1,s2))
points = np.hstack((S,r))
X,Y = np.array(points[:,1]),np.array(points[:,2])
X,Y = X.reshape(n,1), Y.reshape(n,1)

# 초기값 세팅: 원래는 PhaseⅠ Method로 찾아야 함
mu = np.ones((n,1)) # mu는 lagrangian multiplier
a = random.uniform(-2,-0.5)
b = random.uniform(0.75,1.25)
c = random.uniform(-0.5,0.5)
w = np.vstack((a,b,c,mu)) # w=[a,b,c,mu]^T를 매 스텝마다 업데이트
t = 1

# Dual Interior Point Method
for k in range(100):
    # SVM 라그랑지안 식: L(x,mu) = 1/2[a,b]@[a,b]^T + sigma(mu_i*g_i)
    # Dual IPM: delta f + mu@delta g = 0, mu*g + t = 0 (t->0)
    # 이를 만족하는 a,b,c,mu 를 Newton Method 로 찾아야함

    t = 0.9*t #t->0
    a,b,c,mu = w[0],w[1],w[2],w[3:]
    print(a, b, c)

    # g(w): g_i = si(c-ax-by)+1
    g = c - np.array((a, b)).T @ np.vstack((X.T, Y.T))
    g = S * g.T + 1

    # R(w) 계산
    dg_dx = np.hstack((-S*X,-S*Y,S))
    R1 = np.vstack((a,b,0))+dg_dx.T @ mu
    R2 = mu*g+t*np.ones((n,1))
    R = np.vstack((R1,R2))

    # dR/dw 계산
    dR_dw11 = np.array([[1,0,0],[0,1,0],[0,0,0]])
    dR_dw12 = dg_dx.T
    dR_dw21 = np.diag(np.squeeze(mu))@dg_dx
    dR_dw22 = np.diag(np.squeeze(g))
    dR_dw = np.block([[dR_dw11,dR_dw12],[dR_dw21,dR_dw22]])

    # Newton method 로 w 업데이트, 이때 learning rate 0.5로 설정
    # Backtracking line search 로 learning rate 구해 더 빠르게 최적화 시킬수도 있음
    w = w-0.5*np.linalg.inv(dR_dw)@R

    # 결과 plot 하기: 원래 점들, 결정 경계, 마진
    xmin, xmax, ymin, ymax = -5, 10, -5, 10
    plt.grid()
    plt.scatter(X, Y)
    plt.axis([xmin, xmax, ymin, ymax])
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c / b),'r')
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c + 1 / b),'b')
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c - 1 / b),'b')
    plt.draw();plt.pause(0.05);plt.clf()
