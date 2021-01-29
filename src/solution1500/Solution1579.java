package solution1500;

public class Solution1579 {

    /**
     * 1579. 保证图可完全遍历
     * @param n
     * @param edges
     * @return
     */
    public int maxNumEdgesToRemove(int n, int[][] edges) {
        int ans = 0;
        int[] parentA = new int[n];
        int[] parentB = new int[n];
        // 初始化parent数组
        for (int i = 0; i < n; i++) {
            parentA[i] = i;
            parentB[i] = i;
        }
        // 节点编号改为从 0 开始
        for (int[] edge : edges) {
            --edge[1];
            --edge[2];
        }
        //公共边
        for (int[] edge : edges) {
            if (edge[0] == 3) {
                if (!unify(parentA, edge[1], edge[2])) {
                    ans++;
                } else {
                    unify(parentB, edge[1], edge[2]);
                }
            }
        }
        // 独占边
        for (int[] edge : edges) {
            if (edge[0] == 1) {
                // A独占边
                if (!unify(parentA, edge[1], edge[2])) {
                    ans++;
                }
            } else if (edge[0] == 2) {
                // B独占边
                if (!unify(parentB, edge[1], edge[2])) {
                    ans++;
                }
            }
        }
        // 检查是否无法完全遍历
        int temp = parentA[0];
        for (int a : parentA) {
            if (a != temp) {
                return -1;
            }
        }
        temp = parentB[0];
        for (int b : parentB) {
            if (b != temp) {
                return -1;
            }
        }
        return ans;
    }

    private int find(int[] parent, int a) {
        if (parent[a] != a) {
            parent[a] = find(parent, parent[a]);
        }
        return parent[a];
    }

    private boolean unify(int[] parent, int a, int b) {
        int rootA = find(parent, a);
        int rootB = find(parent, b);
        if (rootA == rootB) {
            return false;
        }
        parent[rootA] = rootB;
        return true;
    }
}
