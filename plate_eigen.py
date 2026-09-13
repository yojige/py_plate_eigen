import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. パラメータ設定
# ==========================================
# 形状情報
Lx = 10.0         # 横の長さ [m]
Ly = 10.0         # 縦の長さ [m]
h  = 0.3         # 平板の厚さ [m]

# 物性値
E     = 2.5e10  # ヤング率 [N/m^2]
#G     = 1.0e10   # せん断弾性係数 [N/m^2]
nu    = 0.167      # ポアソン比
G     = E/(2.0*(1.0+nu))  # せん断弾性係数 [N/m^2]
gamma = 24000.0  # 単位体積重量 [N/m^3]
kappa = 0.833    # せん断形状係数 (RC: 0.833)
g     = 9.80665  # 重力加速度 [m/s^2]
rho   = gamma / g  # 密度 [kg/m^3]
#E     = 2.0 * G * (1.0 + nu)  # ヤング率 [N/m^2]
print(G)

# 要素分割情報
nx = 10          # 横方向の要素分割数
ny = 10          # 縦方向の要素分割数

# 境界条件 ('fixed': 固定, 'pinned': ピン支持)
# 順序: bottom (y=0), top (y=Ly), left (x=0), right (x=Lx)
bc = {
    'bottom': 'pinned',
    'top':    'pinned',
    'left':   'pinned',
    'right':  'pinned'
}

# ==========================================
# 2. メッシュ生成
# ==========================================
n_nodes_x = nx + 1
n_nodes_y = ny + 1
n_nodes = n_nodes_x * n_nodes_y
n_elements = nx * ny

# 節点座標
coords = np.zeros((n_nodes, 2))
for j in range(n_nodes_y):
    for i in range(n_nodes_x):
        nid = j * n_nodes_x + i
        coords[nid] = [i * (Lx / nx), j * (Ly / ny)]

# 要素コネクティビティ (反時計回り: 1->2->3->4)
elements = np.zeros((n_elements, 4), dtype=int)
for j in range(ny):
    for i in range(nx):
        eid = j * nx + i
        n1 = j * n_nodes_x + i
        n2 = n1 + 1
        n3 = (j + 1) * n_nodes_x + (i + 1)
        n4 = (j + 1) * n_nodes_x + i
        elements[eid] = [n1, n2, n3, n4]

# ==========================================
# 3. 材料剛性マトリクス
# ==========================================
# 曲げ剛性マトリクス Db (曲げモーメント - 曲率)
D_const = (E * (h**3)) / (12.0 * (1.0 - nu**2))
Db = D_const * np.array([
    [1.0, nu,  0.0],
    [nu,  1.0, 0.0],
    [0.0, 0.0, (1.0 - nu) / 2.0]
])

# せん断剛性マトリクス Ds (せん断力 - せん断ひずみ)
Ds = kappa * G * h * np.eye(2)

# ==========================================
# 4. 単一要素の剛性・質量マトリクス
# ==========================================
def get_element_matrices(x_elem, y_elem):
    ke = np.zeros((12, 12))
    me = np.zeros((12, 12))

    # Gauss 積分点・重み
    gp_2x2 = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    w_2x2  = [1.0, 1.0]

    # --- 1) 曲げ剛性 Ke_b (2x2 ガウス積分) ---
    for xi, wxi in zip(gp_2x2, w_2x2):
        for eta, weta in zip(gp_2x2, w_2x2):
            dN_dxi = 0.25 * np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi),  (1 + xi),  (1 - xi)])
            
            J = np.array([
                [np.dot(dN_dxi, x_elem),  np.dot(dN_dxi, y_elem)],
                [np.dot(dN_deta, x_elem), np.dot(dN_deta, y_elem)]
            ])
            detJ = la.det(J)
            invJ = la.inv(J)

            dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

            Bb = np.zeros((3, 12))
            for a in range(4):
                # 自由度配置: [w_a, theta_x_a, theta_y_a]
                Bb[0, 3 * a + 1] = dN_dx[a]
                Bb[1, 3 * a + 2] = dN_dy[a]
                Bb[2, 3 * a + 1] = dN_dy[a]
                Bb[2, 3 * a + 2] = dN_dx[a]

            ke += (Bb.T @ Db @ Bb) * detJ * wxi * weta

    # --- 2) せん断剛性 Ke_s (1x1 ガウス積分: せん断ロッキング防止) ---
    xi_s, eta_s = 0.0, 0.0
    w_s = 4.0
    dN_dxi = 0.25 * np.array([-(1 - eta_s),  (1 - eta_s),  (1 + eta_s), -(1 + eta_s)])
    dN_deta = 0.25 * np.array([-(1 - xi_s), -(1 + xi_s),  (1 + xi_s),  (1 - xi_s)])
    J = np.array([
        [np.dot(dN_dxi, x_elem),  np.dot(dN_dxi, y_elem)],
        [np.dot(dN_deta, x_elem), np.dot(dN_deta, y_elem)]
    ])
    detJ = la.det(J)
    invJ = la.inv(J)

    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    N_s = 0.25 * np.array([
        (1 - xi_s) * (1 - eta_s),
        (1 + xi_s) * (1 - eta_s),
        (1 + xi_s) * (1 + eta_s),
        (1 - xi_s) * (1 + eta_s)
    ])

    Bs = np.zeros((2, 12))
    for a in range(4):
        Bs[0, 3 * a + 0] = dN_dx[a]
        Bs[0, 3 * a + 1] = -N_s[a]
        Bs[1, 3 * a + 0] = dN_dy[a]
        Bs[1, 3 * a + 2] = -N_s[a]

    ke += (Bs.T @ Ds @ Bs) * detJ * w_s

    # --- 3) 質量マトリクス Me (2x2 ガウス積分) ---
    I_rot = rho * (h**3) / 12.0
    m_trans = rho * h
    for xi, wxi in zip(gp_2x2, w_2x2):
        for eta, weta in zip(gp_2x2, w_2x2):
            N = 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])
            dN_dxi = 0.25 * np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi),  (1 + xi),  (1 - xi)])
            J = np.array([
                [np.dot(dN_dxi, x_elem),  np.dot(dN_dxi, y_elem)],
                [np.dot(dN_deta, x_elem), np.dot(dN_deta, y_elem)]
            ])
            detJ = la.det(J)

            Nw = np.zeros((1, 12))
            Ntx = np.zeros((1, 12))
            Nty = np.zeros((1, 12))
            for a in range(4):
                Nw[0, 3 * a + 0] = N[a]
                Ntx[0, 3 * a + 1] = N[a]
                Nty[0, 3 * a + 2] = N[a]

            me += (m_trans * (Nw.T @ Nw) + I_rot * (Ntx.T @ Ntx + Nty.T @ Nty)) * detJ * wxi * weta

    return ke, me

# ==========================================
# 5. 全体マトリクスの組み立て
# ==========================================
total_dof = 3 * n_nodes
K_global = np.zeros((total_dof, total_dof))
M_global = np.zeros((total_dof, total_dof))

for elem in elements:
    x_e = coords[elem, 0]
    y_e = coords[elem, 1]
    ke, me = get_element_matrices(x_e, y_e)

    edof = []
    for node in elem:
        edof.extend([3 * node, 3 * node + 1, 3 * node + 2])
    
    for r in range(12):
        for c in range(12):
            K_global[edof[r], edof[c]] += ke[r, c]
            M_global[edof[r], edof[c]] += me[r, c]

# ==========================================
# 6. 境界条件の適用
# ==========================================
fixed_dofs = set()
tol = 1e-6

for i, (x, y) in enumerate(coords):
    # Bottom: y = 0
    if abs(y) < tol:
        if bc['bottom'] == 'pinned':
            fixed_dofs.update([3 * i, 3 * i + 1])          # w, theta_x
        elif bc['bottom'] == 'fixed':
            fixed_dofs.update([3 * i, 3 * i + 1, 3 * i + 2]) # w, theta_x, theta_y
    # Top: y = Ly
    if abs(y - Ly) < tol:
        if bc['top'] == 'pinned':
            fixed_dofs.update([3 * i, 3 * i + 1])
        elif bc['top'] == 'fixed':
            fixed_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])
    # Left: x = 0
    if abs(x) < tol:
        if bc['left'] == 'pinned':
            fixed_dofs.update([3 * i, 3 * i + 2])          # w, theta_y
        elif bc['left'] == 'fixed':
            fixed_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])
    # Right: x = Lx
    if abs(x - Lx) < tol:
        if bc['right'] == 'pinned':
            fixed_dofs.update([3 * i, 3 * i + 2])
        elif bc['right'] == 'fixed':
            fixed_dofs.update([3 * i, 3 * i + 1, 3 * i + 2])

all_dofs = np.arange(total_dof)
active_dofs = np.setdiff1d(all_dofs, list(fixed_dofs))

K_act = K_global[np.ix_(active_dofs, active_dofs)]
M_act = M_global[np.ix_(active_dofs, active_dofs)]

# ==========================================
# 7. 固有値解析 & 刺激係数計算
# ==========================================
eigenvalues, eigenvectors = la.eigh(K_act, M_act)

# ソートと周波数・周期の算出
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 面外方向の単位加振ベクトル（w自由度に1.0）
r_vec = np.zeros(total_dof)
for n in range(n_nodes):
    r_vec[3 * n] = 1.0
r_act = r_vec[active_dofs]

print("=== 解析結果 (1次〜5次モード) ===")
print(f"{'Mode':<6} {'Freq [Hz]':<12} {'Period [s]':<12} {'Part. Factor (β)':<16}")

modes_w = []
results_info = []  # プロット表示用に各モードの指標を保存

for m in range(5):
    omega2 = eigenvalues[m]
    omega = np.sqrt(max(0.0, omega2))
    freq = omega / (2.0 * np.pi)
    period = 1.0 / freq if freq > 1e-6 else np.inf
    
    phi_act = eigenvectors[:, m]
    # モード正規化: phi.T @ M @ phi = 1
    modal_mass = phi_act.T @ M_act @ phi_act
    phi_act = phi_act / np.sqrt(modal_mass)
    
    # 刺激係数 beta = (phi.T @ M @ r) / (phi.T @ M @ phi)
    beta = (phi_act.T @ M_act @ r_act) / (phi_act.T @ M_act @ phi_act)
    
    print(f"{m+1:<6} {freq:<12.3f} {period:<12.4f} {beta:<16.4e}")

    # プロット用データ格納
    results_info.append({
        'freq': freq,
        'period': period,
        'beta': beta
    })

    # 全体自由度へのマッピングと変位成分 w の抽出
    phi_full = np.zeros(total_dof)
    phi_full[active_dofs] = phi_act
    modes_w.append(phi_full[0::3])

# ==========================================
# 8. 3D モード図のプロット
# ==========================================
X = coords[:, 0].reshape((n_nodes_y, n_nodes_x))
Y = coords[:, 1].reshape((n_nodes_y, n_nodes_x))

fig = plt.figure(figsize=(16, 9))

for m in range(5):
    ax = fig.add_subplot(2, 3, m + 1, projection='3d')
    
    Z = modes_w[m].reshape((n_nodes_y, n_nodes_x))
    # 見た目のため振幅を最大値1にスケーリング
    max_amp = np.max(np.abs(Z))
    Z_plot = Z / max_amp if max_amp > 1e-9 else Z

    surf = ax.plot_surface(X, Y, Z_plot, cmap='coolwarm', edgecolor='k', linewidth=0.4, alpha=0.85)
    
    # 固有値情報の取得
    f_val = results_info[m]['freq']
    T_val = results_info[m]['period']
    beta_val = results_info[m]['beta']
    
    # タイトルに周波数・周期・刺激係数を改行して表示
    title_text = (
        f"Mode {m+1}\n"
        f"f = {f_val:.2f} Hz  |  T = {T_val:.4f} s\n"
        f"β = {beta_val:+.3e}"
    )
    ax.set_title(title_text, fontsize=10, pad=10)
    
    ax.set_xlabel("X [m]", labelpad=5)
    ax.set_ylabel("Y [m]", labelpad=5)
    ax.set_zlabel("Normalized Shape", labelpad=5)
    ax.view_init(elev=30, azim=-60)

plt.subplots_adjust(hspace=0.35, wspace=0.2)
plt.show()