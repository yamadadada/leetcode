import common.ListNode;
import common.TreeNode;

import java.math.BigInteger;
import java.util.*;

public class Solution {

    /**
     * 31. 下一个排列
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int max_j = -1;
        int max_i = -1;
        for (int i = nums.length - 1; i > 0; --i) {
            for (int j = i - 1; j >= 0; --j) {
                if (nums[i] > nums[j]) {
                    if (j > max_j) {
                        max_j = j;
                        max_i = i;
                    }
                }
            }
        }
        if (max_j != -1) {
            int temp = nums[max_i];
            nums[max_i] = nums[max_j];
            nums[max_j] = temp;
            Arrays.sort(nums, max_j + 1, nums.length);
        } else {
            Arrays.sort(nums);
        }
    }

    /**
     * 36. 有效的数独
     * @param board
     * @return
     */
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[9][9];
        boolean[][] col = new boolean[9][9];
        boolean[][] block = new boolean[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '1';
                    int blockNum = i / 3 * 3 + j / 3;
                    if (row[i][num] || col[j][num] || block[blockNum][num]) {
                        return false;
                    }
                    row[i][num] = true;
                    col[j][num] = true;
                    block[blockNum][num] = true;
                }
            }
        }
        return true;
    }

    /**
     * 48. 旋转图像
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix.length == 0) {
            return;
        }
        int n = matrix[0].length;
        for (int j = 0; j < n / 2; j++) {
            for (int i = j; i < n - 1 - j; i++) {
                int temp = matrix[j][i];
                matrix[j][i] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = matrix[n - 1 - j][n - 1 - i];
                matrix[n - 1 - j][n - 1 - i] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = temp;
            }
        }
    }

    /**
     * 49. 字母异位词分组
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        for (String s: strs) {
            int[] count = new int[26];
            char[] ch = s.toCharArray();
            for (char c: ch) {
                count[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; i++) {
                sb.append("#");
                sb.append(count[i]);
            }
            String key = sb.toString();
            if (map.containsKey(key)) {
                map.get(key).add(s);
            } else {
                List<String> stringList = new ArrayList<>();
                stringList.add(s);
                map.put(key, stringList);
            }
        }
        return new ArrayList<>(map.values());
    }

    /**
     * 55. 跳跃游戏
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums.length == 0) {
            return false;
        }
        if (nums.length == 1) {
            return true;
        }
        int max = 0;
        for (int i = 0; i <= max; i++) {
            if (nums[i] + i > max) {
                max = nums[i] + i;
            }
            if (max >= nums.length - 1) {
                return true;
            }
        }
        return false;
    }

    /**
     * 56. 合并区间
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][];
        }
        List<Interval> list = new ArrayList<>();
        for (int[] is: intervals) {
            list.add(new Interval(is[0], is[1]));
        }
        list.sort(new IntervalComparator());
        List<Interval> ans = new ArrayList<>();
        ans.add(list.get(0));
        int p = 0;
        for (int i = 1; i < list.size(); i++) {
            if (ans.get(p).end >= list.get(i).start) {
                ans.get(p).end = Math.max(list.get(i).end, ans.get(p).end);
            } else {
                ans.add(list.get(i));
                p++;
            }
        }
        int[][] ansArrays = new int[ans.size()][];
        for (int i = 0; i < ans.size(); i++) {
            int[] array = new int[2];
            array[0] = ans.get(i).start;
            array[1] = ans.get(i).end;
            ansArrays[i] = array;
        }
        return ansArrays;
    }

    private static class IntervalComparator implements Comparator<Interval> {
        @Override
        public int compare(Interval o1, Interval o2) {
            return Integer.compare(o1.start, o2.start);
        }
    }

    private static class Interval {
        public int start;

        public int end;

        public Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    /**
     * 64. 最小路径和
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int[][] dp = new int[n][m];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < n; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
            for (int j = 1; j < m; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[n - 1][m - 1];
    }

    /**
     * 73. 矩阵置零
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        if (matrix.length == 0) {
            return;
        }
        boolean row = false;
        boolean column = false;
        // 遍历， 对各列各行的第一个元素标记为0
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    if (i == 0) {
                        row = true;
                    }
                    if (j == 0) {
                        column = true;
                    }
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        // 利用标记的第一个元素对需要的置0
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < matrix[i].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int i = 1; i < matrix[0].length; i++) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < matrix.length; j++) {
                    matrix[j][i] = 0;
                }
            }
        }
        if (column) {
            for (int i = 1; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
        if (row) {
            for (int i = 1; i < matrix[0].length; i++) {
                matrix[0][i] = 0;
            }
        }
    }

    /**
     * 75. 颜色分类
     * @param nums
     */
    public void sortColors(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        int cur = 0;
        while (cur <= r && l < r) {
            if (nums[cur] == 0 && cur > l) {
                int temp = nums[cur];
                nums[cur] = nums[l];
                nums[l] = temp;
                l++;
                cur--;
            } else if (nums[cur] == 2 && cur < r) {
                int temp = nums[cur];
                nums[cur] = nums[r];
                nums[r] = temp;
                r--;
                cur--;
            }
            cur++;
        }
    }

    /**
     * 79. 单词搜索
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (word.length() == 0) {
            return true;
        }
        int n = board.length;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                char c = word.charAt(0);
                if (board[i][j] == c) {
                    boolean[][] visit = new boolean[n][board[0].length];
                    if (dfs(board, word, visit, i, j)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, boolean[][] visit, int i, int j) {
        if (board[i][j] != word.charAt(0)) {
            return false;
        }
        if (word.length() == 1) {
            return true;
        }
        visit[i][j] = true;
        if (i - 1 >= 0 && !visit[i - 1][j] && dfs(board, word.substring(1), visit, i - 1, j)) {
            return true;
        }
        if (i + 1 < board.length && !visit[i + 1][j] && dfs(board, word.substring(1), visit, i + 1, j)) {
            return true;
        }
        if (j - 1 >= 0 && !visit[i][j - 1] && dfs(board, word.substring(1), visit, i, j - 1)) {
            return true;
        }
        if (j + 1 < board[0].length && !visit[i][j + 1] && dfs(board, word.substring(1), visit, i, j + 1)) {
            return true;
        }
        visit[i][j] = false;
        return false;
    }

    /**
     * 83. 删除排序链表中的重复元素
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode pre = head;
        ListNode p = head.next;
        while (p != null) {
            if (pre.val == p.val) {
                pre.next = p.next;
            } else {
                pre = p;
            }
            p = p.next;
        }
        return head;
    }

    /**
     * 91. 解码方法
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        int n = s.length();
        int[] dp = new int[n];
        if (n >= 1 && !s.substring(0, 1).equals("0")) {
            dp[0] = 1;
            if (n >= 2) {
                if (!s.substring(1, 2).equals("0")) {
                    dp[1]++;
                }
                Integer i = Integer.valueOf(s.substring(0, 2));
                if (i <= 26) {
                    dp[1]++;
                }
            }
        }
        for (int i = 2; i < n; i++) {
            int a = Integer.parseInt(s.substring(i - 1, i));
            int b = Integer.parseInt(s.substring(i, i + 1));
            if (a == 0 && b == 0) {
                dp[i] = 0;
            } else if (a == 0) {
                dp[i] = dp[i - 1];
            } else if (b == 0) {
                int c = a * 10 + b;
                if (c <= 26) {
                    dp[i] = dp[i - 2];
                } else {
                    dp[i] = 0;
                }
            } else {
                int c = a * 10 + b;
                dp[i] += dp[i - 1];
                if (c <= 26) {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[n - 1];
    }

    /**
     * 94. 二叉树的中序遍历
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            ans.add(curr.val);
            curr = curr.right;
        }
        return ans;
    }

    /**
     * 98. 验证二叉搜索树
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (isValidBST(root.left)) {
            if (minValue < root.val) {
                minValue = root.val;
                return isValidBST(root.right);
            }
        }
        return false;
    }

    Integer minValue = Integer.MIN_VALUE;

    /**
     * 100. 相同的树
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    /**
     * 101. 对称二叉树
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        if (t1.val != t2.val) {
            return false;
        }
        return isSymmetric(t1.left, t2.right) && isSymmetric(t1.right, t2.left);
    }

    /**
     * 107. 二叉树的层次遍历 II
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int num = queue.size();
            List<Integer> item = new ArrayList<>();
            for (int i = 0; i < num; i++) {
                TreeNode p = queue.poll();
                if (p != null) {
                    item.add(p.val);
                    queue.add(p.left);
                    queue.add(p.right);
                }
            }
            if (item.size() > 0) {
                result.add(0, item);
            }
        }
        return result;
    }

    /**
     * 108. 将有序数组转换为二叉搜索树
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        return separation(nums, 0, nums.length - 1);
    }

    private TreeNode separation(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode treeNode = new TreeNode(nums[mid]);
        treeNode.left = separation(nums, left, mid - 1);
        treeNode.right = separation(nums, mid + 1, right);
        return treeNode;
    }

    /**
     * 109. 有序链表转换二叉搜索树
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        ListNode p = head;
        List<Integer> list = new ArrayList<>();
        while (p != null) {
            list.add(p.val);
            p = p.next;
        }
        return toBST(list, 0, list.size() - 1);
    }

    private TreeNode toBST(List<Integer> list, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(list.get(mid));
        root.left = toBST(list, left, mid - 1);
        root.right = toBST(list, mid + 1, right);
        return root;
    }

    /**
     * 110. 平衡二叉树
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        return height(root) >= 0;
    }

    private int height(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    /**
     * 112. 路径总和
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && sum == root.val) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    /**
     * 114. 二叉树展开为链表
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return;
        }
        flatten(root.left);
        flatten(root.right);
        if (root.left != null) {
            TreeNode t = root.left;
            while (t.right != null) {
                t = t.right;
            }
            t.right = root.right;
            root.right = root.left;
            root.left = null;
        }
    }

    /**
     * 118. 杨辉三角
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        ans.add(Collections.singletonList(1));
        for (int i = 1; i < numRows; i++) {
            List<Integer> list = new ArrayList<>();
            list.add(1);
            List<Integer> previous = ans.get(i - 1);
            for (int j = 1; j < i; j++) {
                list.add(previous.get(j - 1) + previous.get(j));
            }
            list.add(1);
            ans.add(list);
        }
        return ans;
    }

    /**
     * 124. 二叉树中的最大路径和
     * @param root
     * @return
     */
    public int maxPathSum(TreeNode root) {
        getMaxPathSum(root);
        return maxPath;
    }

    private int maxPath = Integer.MIN_VALUE;

    private int getMaxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int a = getMaxPathSum(root.left);
        int b = getMaxPathSum(root.right);
        maxPath = Math.max(maxPath, a + b + root.val);
        a = Math.max(Math.max(a, b), 0);
        maxPath = Math.max(maxPath, a + root.val);
        return a + root.val;
    }

    /**
     * 142. 环形链表 II
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                slow = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }

    /**
     * 167. 两数之和 II - 输入有序数组
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        int i = 0;
        int j = numbers.length - 1;
        while (i < j) {
            int a = numbers[i] + numbers[j];
            if (a == target) {
                result[0] = i + 1;
                result[1] = j + 1;
                return result;
            }
            if (a > target) {
                j--;
            } else {
                i++;
            }
        }
        return result;
    }

    /**
     * 168. Excel表列名称
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while (n > 26) {
            if (n % 26 == 0) {
                sb.append('Z');
                n /= 26;
                n -= 1;
            } else {
                sb.append((char)('A' + (n % 26 - 1)));
                n /= 26;
            }
        }
        sb.append((char)('A' + n - 1));
        return sb.reverse().toString();
    }

    /**
     * 171. Excel表列序号
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        int sum = 0;
        for (int i = 0; i < s.length(); ++i) {
            sum *= 26;
            sum += s.charAt(i) - 'A' + 1;
        }
        return sum;
    }

    /**
     * 238. 除自身以外数组的乘积
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        int left = 1;
        int right = 1;
        int[] output = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            output[i] = left;
            left *= nums[i];
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            output[i] *= right;
            right *= nums[i];
        }
        return output;
    }

    /**
     * 264. 丑数 II
     * @param n
     * @return
     */
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int i2 = 0;
        int i3 = 0;
        int i5 = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = Math.min(dp[i2] * 2, Math.min(dp[i3] * 3, dp[i5] * 5));
            if (dp[i] == dp[i2] * 2) {
                i2++;
            }
            if (dp[i] == dp[i3] * 3) {
                i3++;
            }
            if (dp[i] == dp[i5] * 5) {
                i5++;
            }
        }
        return dp[n - 1];
    }

    /**
     * 329. 矩阵中的最长递增路径
     * @param matrix
     * @return
     */
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix.length == 0) {
            return 0;
        }
        int[][] a = new int[matrix.length][matrix[0].length];
        int max = 0;
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                max = Math.max(max, recursion(matrix, a, i, j));
            }
        }
        return max;
    }

    private int recursion(int[][] matrix, int[][] a, int i, int j) {
        // 已有数据，不重复遍历
        if (a[i][j] != 0) {
            return a[i][j];
        }
        int max = 1;
        // 搜索四周，寻找比自己大的
        if (j + 1 < matrix[0].length && matrix[i][j + 1] > matrix[i][j]) {
            max = Math.max(max, recursion(matrix, a, i, j + 1) + 1);
        }
        if (i + 1 < matrix.length && matrix[i + 1][j] > matrix[i][j]) {
            max = Math.max(max, recursion(matrix, a, i + 1, j) + 1);
        }
        if (j - 1 >= 0 && matrix[i][j - 1] > matrix[i][j]) {
            max = Math.max(max, recursion(matrix, a, i, j - 1) + 1);
        }
        if (i - 1 >= 0 && matrix[i - 1][j] > matrix[i][j]) {
            max = Math.max(max, recursion(matrix, a, i - 1, j) + 1);
        }
        a[i][j] = max;
        return max;
    }

    /**
     * 338. 比特位计数
     * @param num
     * @return
     */
    public int[] countBits(int num) {
        int[] result = new int[num + 1];
        result[0] = 0;
        if (num == 0) {
            return result;
        }
        result[1] = 1;
        int a = 2;
        int k = 0;
        for (int i = 2; i <= num; i++) {
            if (k == a) {
                a *= 2;
                k = 0;
            }
            result[i] = result[i - a] + 1;
            k++;
        }
        return result;
    }

    /**
     * 443. 压缩字符串
     * @param chars
     * @return
     */
    public int compress(char[] chars) {
        int i = 1;
        int charIndex = 0;
        int countIndex = 1;
        int count = 1;
        while (i < chars.length) {
            if (chars[i] != chars[i - 1]) {
                chars[charIndex] = chars[i - 1];
                if (count > 1) {
                    String s = String.valueOf(count);
                    for (int j = 0; j < s.length(); j++) {
                        chars[countIndex + j] = s.charAt(j);

                    }
                    charIndex = countIndex + s.length();
                } else {
                    charIndex = countIndex;
                }
                countIndex = charIndex + 1;
                count = 0;
            }
            count++;
            i++;
        }
        chars[charIndex] = chars[i - 1];
        if (count > 1) {
            String s = String.valueOf(count);
            for (int j = 0; j < s.length(); j++) {
                chars[countIndex + j] = s.charAt(j);
            }
            charIndex = countIndex + s.length();
        } else {
            charIndex = countIndex;
        }
        return charIndex;
    }

    /**
     * 455. 分发饼干
     * @param g
     * @param s
     * @return
     */
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0;
        int j = 0;
        int count = 0;
        while (i < g.length && j < s.length) {
            if (g[i] <= s[j++]) {
                count++;
                i++;
            }
        }
        return count;
    }

    /**
     * 529. 扫雷游戏
     * @param board
     * @param click
     * @return
     */
    public char[][] updateBoard(char[][] board, int[] click) {
        if (board[click[0]][click[1]] == 'M') {
            board[click[0]][click[1]] = 'X';
            return board;
        }
        // 如果挖出的是E
        int count = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (isMine(board, click[0] + i, click[1] + j)) {
                    ++count;
                }
            }
        }
        if (count == 0) {
            board[click[0]][click[1]] = 'B';
        } else {
            // ASCII 48 为 ‘0’
            board[click[0]][click[1]] = (char)(count + 48);
        }
        // 递归开挖E，只有为空方块的时候才向四周开挖
        if (board[click[0]][click[1]] == 'B') {
            if (click[0] - 1 >= 0 && board[click[0] - 1][click[1]] == 'E') {
                updateBoard(board, new int[]{click[0] - 1, click[1]});
            }
            if (click[0] + 1 < board.length && board[click[0] + 1][click[1]] == 'E') {
                updateBoard(board, new int[]{click[0] + 1, click[1]});
            }
            if (click[1] - 1 >= 0 && board[click[0]][click[1] - 1] == 'E') {
                updateBoard(board, new int[]{click[0], click[1] - 1});
            }
            if (click[1] + 1 < board[0].length && board[click[0]][click[1] + 1] == 'E') {
                updateBoard(board, new int[]{click[0], click[1] + 1});
            }
            if (click[0] - 1 >= 0 && click[1] - 1 >= 0 && board[click[0] - 1][click[1] - 1] == 'E') {
                updateBoard(board, new int[]{click[0] - 1, click[1] - 1});
            }
            if (click[0] - 1 >= 0 && click[1] + 1 < board[0].length && board[click[0] - 1][click[1] + 1] == 'E') {
                updateBoard(board, new int[]{click[0] - 1, click[1] + 1});
            }
            if (click[0] + 1 < board.length && click[1] - 1 >= 0 && board[click[0] + 1][click[1] - 1] == 'E') {
                updateBoard(board, new int[]{click[0] + 1, click[1] - 1});
            }
            if (click[0] + 1 < board.length && click[1] + 1 < board[0].length && board[click[0] + 1][click[1] + 1] == 'E') {
                updateBoard(board, new int[]{click[0] + 1, click[1] + 1});
            }
        }
        return board;
    }

    // 判断是否是地雷，同时对x，y做界限校验
    private boolean isMine(char[][] board, int x, int y) {
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) {
            return false;
        }
        return board[x][y] == 'M' || board[x][y] == 'X';
    }

    /**
     * 539. 最小时间差
     * @param timePoints
     * @return
     */
    public int findMinDifference(List<String> timePoints) {
        List<Integer> list = new ArrayList<>();
        for (String s : timePoints) {
            String[] array = s.split(":");
            list.add(Integer.parseInt(array[0]) * 60 + Integer.parseInt(array[1]));
        }
        Collections.sort(list);
        if (list.size() > 0) {
            list.add(list.get(0) + 24 * 60);
        }
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < list.size(); i++) {
            min = Math.min(min, list.get(i) - list.get(i - 1));
        }
        return min;
    }

    /**
     * 546. 移除盒子
     * @param boxes
     * @return
     */
    public int removeBoxes(int[] boxes) {
        int[][][] dp = new int[100][100][100];
        return calculatePoints(boxes, dp, 0, boxes.length - 1, 0);
    }

    private int calculatePoints(int[] boxes, int[][][] dp, int l, int r, int k) {
        if (l > r) {
            return 0;
        }
        if (dp[l][r][k] != 0) {
            return dp[l][r][k];
        }
        while (l < r && boxes[r] == boxes[r - 1]) {
            r--;
            k++;
        }
        // 策略一
        dp[l][r][k] = calculatePoints(boxes, dp, l, r - 1, 0) + (k + 1) * (k + 1);
        // 策略二
        for (int i = l; i < r - 1; ++i) {
            if (boxes[i] == boxes[r]) {
                dp[l][r][k] = Math.max(dp[l][r][k], calculatePoints(boxes, dp, l, i, k + 1) + calculatePoints(boxes, dp, i + 1, r - 1, 0));
            }
        }
        return dp[l][r][k];
    }

    /**
     * 617. 合并二叉树
     * @param t1
     * @param t2
     * @return
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return null;
        }
        TreeNode root = new TreeNode(0);
        if (t1 != null) {
            root.val += t1.val;
        }
        if (t2 != null) {
            root.val += t2.val;
        }
        if (t1 != null && t2 != null) {
            root.left = mergeTrees(t1.left, t2.left);
            root.right = mergeTrees(t1.right, t2.right);
        } else if (t1 == null) {
            root.left = mergeTrees(null, t2.left);
            root.right = mergeTrees(null, t2.right);
        } else {
            root.left = mergeTrees(t1.left, null);
            root.right = mergeTrees(t1.right, null);
        }
        return root;
    }

    /**
     * 643. 子数组最大平均数 I
     * @param nums
     * @param k
     * @return
     */
    public double findMaxAverage(int[] nums, int k) {
        if (nums.length < k) {
            return 0;
        }
        int max = 0;
        int sum = 0;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
            max = sum;
        }
        for (int i = k; i < nums.length; i++) {
            sum -= nums[i - k];
            sum += nums[i];
            max = Math.max(max, sum);
        }
        return max / (double) k;
    }

    /**
     * 647. 回文子串
     * @param s
     * @return
     */
    public int countSubstrings(String s) {
        int count = s.length();
        for (int i = 1; i < s.length() - 1; ++i) {
            int j = 1;
            while (i - j >= 0 && i + j < s.length() && s.charAt(i - j) == s.charAt(i + j)) {
                ++count;
                ++j;
            }
        }
        for (int i = 1; i < s.length(); ++i) {
            int j = 0;
            while (i - 1 - j >= 0 && i + j < s.length() && s.charAt(i - 1 - j) == s.charAt(i + j)) {
                ++count;
                ++j;
            }
        }
        return count;
    }

    /**
     * 662. 二叉树最大宽度
     * @param root
     * @return
     */
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int max = 1;
        int leftCount = 0;
        int rightCount = 0;
        TreeNode left = root;
        TreeNode right = root;
        while (left != null && right != null) {
            max = Math.max(max, rightCount - leftCount + 1);
            if (left.left != null) {
                left = left.left;
                leftCount = leftCount * 2 + 1;
            } else {
                left = left.right;
                leftCount = leftCount * 2 + 2;
            }
            if (right.right != null) {
                right = right.right;
                rightCount = rightCount * 2 + 2;
            } else {
                right = right.left;
                rightCount = rightCount * 2 + 1;
            }
        }
        return max;
    }

    /**
     * 712. 两个字符串的最小ASCII删除和
     * @param s1
     * @param s2
     * @return
     */
    public int minimumDeleteSum(String s1, String s2) {
        int n1 = s1.length();
        int n2 = s2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = n1 - 1; i >= 0; i--) {
            dp[i][n2] = dp[i + 1][n2] + s1.codePointAt(i);
        }
        for (int i = n2 - 1; i >= 0; i--) {
            dp[n1][i] = dp[n1][i + 1] + s2.codePointAt(i);
        }
        for (int i = n1 - 1; i >= 0; i--) {
            for (int j = n2 - 1; j >= 0; j--) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    dp[i][j] = dp[i + 1][j + 1];
                } else {
                    dp[i][j] = Math.min(dp[i + 1][j] + s1.codePointAt(i), dp[i][j + 1] + s2.codePointAt(j));
                }
            }
        }
        return dp[0][0];
    }

    /**
     * 713. 乘积小于K的子数组
     * @param nums
     * @param k
     * @return
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) {
            return 0;
        }
        int result = 0;
        int value = 1;
        int left = 0;
        for (int right = 0; right < nums.length; right++) {
            value *= nums[right];
            while (value >= k) {
                value /= nums[left++];
            }
            result += right - left + 1;
        }
        return result;
    }

    /**
     * 733. 图像渲染
     * @param image
     * @param sr
     * @param sc
     * @param newColor
     * @return
     */
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int tempColor = image[sr][sc];
        // 如果已经是新颜色，则不需要遍历修改，否则会造成死循环
        if (tempColor == newColor) {
            return image;
        }
        image[sr][sc] = newColor;
        if (sc + 1 < image[0].length && image[sr][sc + 1] == tempColor) {
            floodFill(image, sr, sc + 1, newColor);
        }
        if (sr + 1 < image.length && image[sr + 1][sc] == tempColor) {
            floodFill(image, sr + 1, sc, newColor);
        }
        if (sc - 1 >= 0 && image[sr][sc - 1] == tempColor) {
            floodFill(image, sr, sc - 1, newColor);
        }
        if (sr - 1 >= 0 && image[sr - 1][sc] == tempColor) {
            floodFill(image, sr - 1, sc, newColor);
        }
        return image;
    }

    /**
     * 763. 划分字母区间
     * @param S
     * @return
     */
    public List<Integer> partitionLabels(String S) {
        if (S.length() == 0) {
            return new ArrayList<>();
        }
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < S.length(); i++) {
            Character c = S.charAt(i);
            map.put(c, i);
        }
        List<Integer> result = new ArrayList<>();
        int i = 0;
        int max = 0;
        int before = 0;
        while (i < S.length()) {
            max = Math.max(max, map.get(S.charAt(i)));
            if (i >= max) {
                result.add(i - before + 1);
                before = i + 1;
            }
            i++;
        }
        return result;
    }

    /**
     * 790. 多米诺和托米诺平铺
     * @param N
     * @return
     */
    public int numTilings(int N) {
        List<BigInteger> list = new ArrayList<>();
        list.add(new BigInteger(String.valueOf(1)));
        list.add(new BigInteger(String.valueOf(1)));
        for (int i = 2; i <= N; i++) {
            BigInteger sum = new BigInteger(String.valueOf(0));
            int sum1;
            for (int j = 1; j <= i; j++) {
                if (j <= 2) {
                    sum1 = 1;
                } else {
                    sum1 = 2;
                }
                BigInteger temp = new BigInteger(String.valueOf(sum1));
                temp = temp.multiply(list.get(i - j));
                sum = sum.add(temp);
            }
            list.add(sum);
        }
        return list.get(N).mod(new BigInteger(String.valueOf(1000000007))).intValue();
    }

    /**
     * 837. 新21点
     * @param N
     * @param K
     * @param W
     * @return
     */
    public double new21Game(int N, int K, int W) {
        int[] dp = new int[K + W];
        return 0;
    }

    /**
     * 859. 亲密字符串
     * @param A
     * @param B
     * @return
     */
    public boolean buddyStrings(String A, String B) {
        if (A.length() != B.length()) {
            return false;
        }
        if (A.equals(B)) {
            int[] count = new int[26];
            for (int i = 0; i < A.length(); i++) {
                count[A.charAt(i) - 'a']++;
            }
            for (int c : count) {
                if (c > 1) {
                    return true;
                }
            }
            return false;
        }
        int first = -1;
        int second = -1;
        for (int i = 0; i < A.length(); i++) {
            if (A.charAt(i) != B.charAt(i)) {
                if (first == -1) {
                    first = i;
                } else if (second == -1) {
                    second = i;
                } else {
                    return false;
                }
            }
        }
        return second != -1 && A.charAt(first) == B.charAt(second) && A.charAt(second) == B.charAt(first);
    }

    /**
     * 872. 叶子相似的树
     * @param root1
     * @param root2
     * @return
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        dfs(root1, list1);
        dfs(root2, list2);
        return list1.equals(list2);
    }

    private void dfs(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            list.add(root.val);
        }
        dfs(root.left, list);
        dfs(root.right, list);
    }

    /**
     * 908. 最小差值 I
     * @param A
     * @param K
     * @return
     */
    public int smallestRangeI(int[] A, int K) {
        int max = A[0];
        int min = A[0];
        for (int i : A) {
            max = Math.max(max, i);
            min = Math.min(min, i);
        }
        return Math.max(0, max - min - 2 * K);
    }

    /**
     * 914. 卡牌分组
     * @param deck
     * @return
     */
    public boolean hasGroupsSizeX(int[] deck) {
        if (deck.length == 0) {
            return false;
        }
        Arrays.sort(deck);
        int count = 0;
        int current = -1;
        List<Integer> countList = new ArrayList<>();
        for (int d : deck) {
            if (d != current) {
                if (count == 1) {
                    return false;
                }
                if (count != 0) {
                    countList.add(count);
                }
                count = 0;
                current = d;
            }
            count++;
        }
        if (count == 1) {
            return false;
        }
        countList.add(count);
        int first = countList.get(0);
        for (int i = 1; i < countList.size(); i++) {
            first = getFactor(countList.get(i), first);
            if (first < 2) {
                return false;
            }
        }
        return true;
    }

    private int getFactor(int a, int b) {
        if (a < b) {
            int temp = a;
            a = b;
            b = temp;
        }
        if (a % b == 0) {
            return b;
        }
        return getFactor(b, a % b);
    }

    /**
     * 929. 独特的电子邮件地址
     * @param emails
     * @return
     */
    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for (String email : emails) {
            String[] array = email.split("@");
            array[0] = array[0].replace(".", "");
            if (array[0].contains("+")) {
                int index = array[0].indexOf("+");
                array[0] = array[0].substring(0, index);
            }
            set.add(array[0] + "@" + array[1]);
        }
        return set.size();
    }

    /**
     * 977. 有序数组的平方
     * @param A
     * @return
     */
    public int[] sortedSquares(int[] A) {
        int index = A.length;
        for (int i = 0; i < A.length; i++) {
            if (A[i] >= 0) {
                index = i;
                break;
            }
        }
        int i = index - 1;
        int j = index;
        int[] B = new int[A.length];
        int k = 0;
        while (i >= 0 && j < A.length) {
            if (A[i] * A[i] <= A[j] * A[j]) {
                B[k++] = A[i] * A[i];
                i--;
            } else {
                B[k++] = A[j] * A[j];
                j++;
            }
        }
        while (i >= 0) {
            B[k++] = A[i] * A[i];
            i--;
        }
        while (j < A.length) {
            B[k++] = A[j] * A[j];
            j++;
        }
        return B;
    }

    /**
     * 1008. 先序遍历构造二叉树
     * @param preorder
     * @return
     */
    public TreeNode bstFromPreorder(int[] preorder) {
        if (preorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[0]);
        int mid = preorder.length;
        for (int i = 1; i < preorder.length; i++) {
            if (preorder[i] > preorder[0]) {
                mid = i;
                break;
            }
        }
        root.left = bstFromPreorder(Arrays.copyOfRange(preorder, 1, mid));
        root.right = bstFromPreorder(Arrays.copyOfRange(preorder, mid, preorder.length));
        return root;
    }

    /**
     * 1019. 链表中的下一个更大节点
     * @param head
     * @return
     */
    public int[] nextLargerNodes(ListNode head) {
        Stack<List<Integer>> stack = new Stack<>();
        ListNode p = head;
        int n = 0;
        while (p != null) {
            n++;
            p = p.next;
        }
        int[] result = new int[n];
        p = head;
        int i = 0;
        while (p != null) {
            while (!stack.isEmpty() && stack.peek().get(0) < p.val) {
                List<Integer> a = stack.pop();
                result[a.get(1)] = p.val;
            }
            List<Integer> temp = new ArrayList<>();
            temp.add(p.val);
            temp.add(i);
            stack.push(temp);
            p = p.next;
            i++;
        }
        return result;
    }

    /**
     * 1022. 从根到叶的二进制数之和
     * @param root
     * @return
     */
    public int sumRootToLeaf(TreeNode root) {
        dfs(root, 0);
        int mod = 1000000007;
        return sum % mod;
    }

    private void dfs(TreeNode root, int val) {
        if (root == null) {
            return;
        }
        val = val << 1 | root.val;
        if (root.left == null && root.right == null) {
            sum += val;
        }
        dfs(root.left, val);
        dfs(root.right, val);
    }

    private int sum;
}
