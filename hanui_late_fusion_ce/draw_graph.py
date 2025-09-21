import matplotlib.pyplot as plt

def draw_mrd():
    x_labels = ["Valid", "T01", "T02", "T03", "T04"]
    y_data = [
        [32.6, 18.8, 25.9],
        [19.7, 43.4, 24.0],
        [32.5, 23.4, 24.1],
        [33.7, 31.4, 30.8],
        [28.3, 52.1, 30.7],
    ]

    # MRD별 데이터
    mrd1 = [row[0] for row in y_data]
    mrd2 = [row[1] for row in y_data]
    mrd3 = [row[2] for row in y_data]

    # 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, mrd1, marker='o', color='blue', linestyle='-', label="MRD 1")
    plt.plot(x_labels, mrd2, marker='s', color='red', linestyle='--', label="MRD 2")
    plt.plot(x_labels, mrd3, marker='^', color='green', linestyle='-.', label="MRD 3")

    plt.xlabel("Test Set")
    plt.ylabel("EER (%)")
    plt.title("EER across Test Sets")
    plt.legend()
    plt.grid(True)
    plt.savefig("mrd.png")

def draw_band():
    # 데이터
    x_labels = ["Valid", "T01", "T02", "T03", "T04"]
    y_data_1 = [
        [28.0, 33.1, 50.6, 52.9, 59.0],
        [30.0, 29.5, 30.2, 32.9, 44.4],
        [32.3, 33.6, 28.5, 53.8, 55.3],
        [33.8, 35.1, 29.3, 55.6, 53.6],
        [28.7, 47.0, 34.1, 50.5, 52.1],
    ]
    y_data_2 = [
        [33.2, 42.5, 43.7, 23.5, 30.5],
        [38.6, 35.9, 51.8, 43.7, 57.3],
        [38.6, 42.6, 52.8, 27.1, 25.6],
        [39.3, 42.9, 52.5, 28.7, 45.6],
        [42.0, 38.5, 56.4, 20.1, 57.2],
    ]
    y_data_3 = [
        [40.9, 41.1, 38.1, 32.9, 34.6],
        [37.7, 43.4, 33.2, 28.0, 42.2],
        [43.3, 48.4, 40.9, 30.1, 28.9],
        [42.9, 47.6, 40.8, 33.2, 44.1],
        [53.6, 42.8, 37.3, 34.2, 55.2],
    ]
    markers = ['o', 's', '^', 'D', 'v']
    colors_1 = ['blue', 'lightblue', 'dodgerblue', 'navy', 'skyblue']
    colors_2 = ['red', 'salmon', 'darkred', 'orange', 'crimson']
    colors_3 = ['green', 'lime', 'darkgreen', 'seagreen', 'olive']

    plt.figure(figsize=(10,6))

    # MRD 1
    for i, row in enumerate(y_data_1):
        plt.plot(x_labels, row, marker=markers[i], color=colors_1[i], linestyle='-', label=f'MRD 1-{i+1}')
    title = 'sub-band_MRD_1.png'

    # # MRD 2
    # for i, row in enumerate(y_data_2):
    #     plt.plot(x_labels, row, marker=markers[i], color=colors_2[i], linestyle='--', label=f'MRD 2-{i+1}')
    # title = 'sub-band_MRD_2.png'

    # # MRD 3
    # for i, row in enumerate(y_data_3):
    #     plt.plot(x_labels, row, marker=markers[i], color=colors_3[i], linestyle='-.', label=f'MRD 3-{i+1}')
    # title = 'sub-band_MRD_3.png'

    plt.xlabel("Test Set")
    plt.ylabel("EER (%)")
    plt.title("EER for sub-bands across Test Sets")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 범례를 그래프 밖으로
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(title)

draw_band()