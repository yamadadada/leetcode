package solution900;

public class Solution959 {

    /**
     * 959. 由斜杠划分区域
     * @param grid
     * @return
     */
    public int regionsBySlashes(String[] grid) {
        int N = grid.length;
        int size = 4 * N * N;
        int[] parent = new int[size];
        for (int i = 0; i < size; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // 单元格内合并
                if (grid[i].charAt(j) == '/') {
                    // 合并0、3 ， 1、2
                    union(parent, 4 * (i * N + j), 4 * (i * N + j) + 3);
                    union(parent, 4 * (i * N + j) + 1, 4 * (i * N + j) + 2);
                } else if (grid[i].charAt(j) == '\\') {
                    // 合并0、1 ， 2、3
                    union(parent, 4 * (i * N + j), 4 * (i * N + j) + 1);
                    union(parent, 4 * (i * N + j) + 2, 4 * (i * N + j) + 3);
                } else {
                    union(parent, 4 * (i * N + j), 4 * (i * N + j) + 1);
                    union(parent, 4 * (i * N + j) + 1, 4 * (i * N + j) + 2);
                    union(parent, 4 * (i * N + j) + 2, 4 * (i * N + j) + 3);
                }
                // 单元格间合并
                if (j < N - 1) {
                    // 向右合并
                    union(parent, 4 * (i * N + j) + 1, 4 * (i * N + j + 1) + 3);
                }
                if (i < N - 1) {
                    // 向下合并
                    union(parent, 4 * (i * N + j) + 2, 4 * ((i + 1) * N + j));
                }
            }
        }
        // 统计集合个数
        int res = 0;
        for (int i = 0; i < parent.length; i++) {
            if (parent[i] == i) {
                res++;
            }
        }
        return res;
    }

    private int find(int[] parent, int a) {
        if (parent[a] != a) {
            parent[a] = find(parent, parent[a]);
        }
        return parent[a];
    }

    private void union(int[] parent, int a, int b) {
        int root1 = find(parent, a);
        int root2 = find(parent, b);
        if (root1 != root2) {
            parent[root2] = root1;
        }
    }
}
