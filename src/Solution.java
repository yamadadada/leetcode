import com.sun.org.apache.bcel.internal.generic.IF_ACMPEQ;
import common.ListNode;
import common.Node;
import common.TreeNode;

import java.math.BigInteger;
import java.util.*;
import java.util.stream.Collectors;

public class Solution {

    /**
     * 17. 电话号码的字母组合
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        String[] phoneMap = new String[]{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> ans = new ArrayList<>();
        if (digits.length() != 0) {
            StringBuilder sb = new StringBuilder();
            phoneLetter(digits, 0, phoneMap, ans, sb);
        }
        return ans;
    }

    /**
     * 递归dfs遍历
     * @param digits
     * @param i 当前递归的位置下标
     * @param phoneMap
     * @param ans
     */
    private void phoneLetter(String digits, int i, String[] phoneMap, List<String> ans, StringBuilder sb) {
        if (i == digits.length()) {
            ans.add(sb.toString());
            return;
        }
        int phone = Integer.parseInt(digits.substring(i, i + 1));
        String letter = phoneMap[phone - 2];
        for (int j = 0; j < letter.length(); j++) {
            sb.append(letter.charAt(j));
            phoneLetter(digits, i + 1, phoneMap, ans, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * 19. 删除链表的倒数第N个节点
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return null;
        }
        ListNode myHead = new ListNode(0);
        myHead.next = head;
        ListNode p = myHead;
        ListNode q = myHead;
        while (p.next != null) {
            p = p.next;
            if (n <= 0) {
                q = q.next;
            }
            n -= 1;
        }
        q.next = q.next.next;
        return myHead.next;
    }

    /**
     * 24. 两两交换链表中的节点
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode b = head.next;
        ListNode c = head.next.next;
        b.next = head;
        head.next = swapPairs(c);
        return b;
    }
    
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
     * 39. 组合总和
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < candidates.length; i++) {
            dfs(candidates, target, ans, new ArrayList<>(), i, 0);
        }
        return ans;
    }

    private void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> list, int i, int sum) {
        sum += candidates[i];
        list.add(candidates[i]);
        if (sum == target) {
            ans.add(new ArrayList<>(list));
        }
        if (sum < target) {
            for (; i < candidates.length; i++) {
                dfs(candidates, target, ans, list, i, sum);
            }
        }
        list.remove(list.size() - 1);
    }

    /**
     * 40. 组合总和 II
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        // 去重
        Set<String> set = new HashSet<>();
        for (int i = 0; i < candidates.length; i++) {
            if (i == 0 || candidates[i - 1] != candidates[i]) {
                dfs(ans, new ArrayList<>(), candidates, target, i, 0, set);
            }
        }
        return ans;
    }

    private void dfs(List<List<Integer>> ans, List<Integer> list, int[] candidates, int target, int i, int sum, Set<String> set) {
        list.add(candidates[i]);
        sum += candidates[i];
        if (sum == target) {
            StringBuilder sb = new StringBuilder();
            for (int temp : list) {
                sb.append(temp).append(",");
            }
            if (!set.contains(sb.toString())) {
                ans.add(new ArrayList<>(list));
                set.add(sb.toString());
            }
        }
        if (sum < target) {
            for (i = i + 1; i < candidates.length; i++) {
                dfs(ans, list, candidates, target, i, sum, set);
            }
        }
        list.remove(list.size() - 1);
    }

    /**
     * 47. 全排列 II
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> ans = new ArrayList<>();
        huisu(nums, ans, new ArrayList<>(), used);
        return ans;
    }

    private void huisu(int[] nums, List<List<Integer>> ans, List<Integer> list, boolean[] used) {
        if (list.size() == nums.length) {
            ans.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i - 1] == nums[i] && !used[i - 1]) {
                continue;
            }
            list.add(nums[i]);
            used[i] = true;
            huisu(nums, ans, list, used);
            list.remove(list.size() - 1);
            used[i] = false;
        }
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
     * 51. N 皇后
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> ans = new ArrayList<>();
        if (n == 1) {
            List<String> list = new ArrayList<>();
            list.add("Q");
            ans.add(list);
            return ans;
        }
        if (n < 4) {
            return ans;
        }
        for (int j = 0; j < n; j++) {
            int[][] array = new int[n][n];
            array[0][j] = 2;
            extendArray(array, 0, j, n);
            dfs(ans, array, 1, n);
        }
        return ans;
    }

    private void dfs(List<List<String>> ans, int[][] array, int i, int n) {
        if (i == n) {
            List<String> list = new ArrayList<>();
            for (int[] b : array) {
                StringBuilder sb = new StringBuilder();
                for (int c : b) {
                    if (c == 2) {
                        sb.append("Q");
                    } else {
                        sb.append(".");
                    }
                }
                list.add(sb.toString());
            }
            ans.add(list);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (array[i][j] == 0) {
                int[][] a = new int[n][n];
                for (int k = 0; k < n; k++) {
                    System.arraycopy(array[k], 0, a[k], 0, n);
                }
                a[i][j] = 2;
                extendArray(a, i, j, n);
                dfs(ans, a, i + 1, n);
            }
        }
    }

    private void extendArray(int[][] a, int i, int j, int n) {
        for (int k = 1; i + k < n; k++) {
            a[i + k][j] = 1;
            if (j - k >= 0) {
                a[i + k][j - k] = 1;
            }
            if (j + k < n) {
                a[i + k][j + k] = 1;
            }
        }
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
     * 57. 插入区间
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int[][] ans = new int[intervals.length][];
//        int k = 0;
//        int i = 0;
//        while (i < intervals.length && intervals[i][1] < newInterval[0]) {
//            int[] tempArray = new int[2];
//            tempArray[0] = intervals[i][0];
//            tempArray[1] = intervals[i][1];
//            ans[k++] = tempArray;
//            i++;
//        }
//        while (i < intervals.length && (intervals[i][1] >= newInterval[0] || (intervals[i][0] <= newInterval[1]))) {
//            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
//            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
//            i++;
//        }
//        int[] tempArray = new int[2];
//        tempArray[0] = newInterval[0];
//        tempArray[1] = newInterval[1];
//        ans[k++] = tempArray;
//        while (i < intervals.length && intervals[i][0] > newInterval[1]) {
//            tempArray = new int[2];
//            tempArray[0] = intervals[i][0];
//            tempArray[1] = intervals[i][1];
//            ans[k++] = tempArray;
//            i++;
//        }
        return ans;
    }

    /**
     * 60. 第k个排列
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n == 1) {
            return "1";
        }
        int factorial = getFactorial(n - 1);
        int a = (k - 1) / factorial + 1;
        int b = (k - 1) % factorial + 1;
        String s = getPermutation(n - 1, b);
        StringBuilder sb = new StringBuilder(a);
        for (int i = 0; i < s.length(); i++) {
            int value = Integer.parseInt(s.substring(i, i + 1));
            if (value >= a) {
                value++;
            }
            sb.append(value);
        }
        return a + sb.toString();
    }

    private int getFactorial(int n) {
        int i = 2;
        int sum = 1;
        while (i <= n) {
            sum *= i;
            i++;
        }
        return sum;
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
     * 77. 组合
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            dfs(ans, list, n, k, i);
        }
        return ans;
    }

    private void dfs(List<List<Integer>> ans, List<Integer> list, int n, int k, int a) {
        list.add(a);
        if (list.size() == k) {
            ans.add(new ArrayList<>(list));
        } else {
            for (int i = a + 1; i <= n; i++) {
                dfs(ans, list, n, k, i);
            }
        }
        list.remove(list.size() - 1);
    }

    /**
     * 78. 子集
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i <= nums.length; i++) {
            huisu(nums, ans, new ArrayList<>(), i);
        }
        return ans;
    }

    private void huisu(int[] nums, List<List<Integer>> ans, List<Integer> list, int i) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(list));
            return;
        }
        list.add(nums[i]);
        for (i = i + 1; i <= nums.length; i++) {
            huisu(nums, ans, list, i);
        }
        list.remove(list.size() - 1);
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
     * 111. 二叉树的最小深度
     * @param root
     * @return
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int d1 = minDepth(root.left);
        int d2 = minDepth(root.right);
        return root.left == null || root.right == null ? d1 + d2 + 1 : Math.min(d1, d2) + 1;
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
     * 113. 路径总和 II
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> ans = new ArrayList<>();
        pathSum2(ans, new ArrayList<>(), root, sum);
        return ans;
    }

    private void pathSum2(List<List<Integer>> ans, List<Integer> list, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        sum -= root.val;
        list.add(root.val);
        if (sum == 0 && root.left == null && root.right == null) {
            ans.add(new ArrayList<>(list));
        }
        pathSum2(ans, list, root.left, sum);
        pathSum2(ans, list, root.right, sum);
        list.remove(list.size() - 1);
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
     * 116. 填充每个节点的下一个右侧节点指针
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node node = queue.poll();
                if (i < size - 1) {
                    node.next = queue.peek();
                }
                if (node.left != null) {
                    queue.add(node.left);
                    queue.add(node.right);
                }
            }
        }
        return root;
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
     * 122. 买卖股票的最佳时机 II
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices.length <= 1) {
            return 0;
        }
        int res = 0;
        int min;
        int max;
        int i = 1;
        while (i < prices.length) {
            while (i < prices.length && prices[i - 1] >= prices[i]) {
                i++;
            }
            min = prices[i - 1];
            max = prices[i - 1];
            while (i < prices.length && prices[i - 1] < prices[i]) {
                max = prices[i];
                i++;
            }
            res += max - min;
        }
        return res;
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
     * 127. 单词接龙
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (beginWord.equals(endWord)) {
            return 0;
        }
        int ans = 1;
        List<List<Integer>> lists = new ArrayList<>();
        int endWordIndex = -1;
        for (int i = 0; i < wordList.size(); i++) {
            String word = wordList.get(i);
            // 检查endWord是否在wordList中
            if (endWord.equals(word)) {
                endWordIndex = i;
            }
            addEdge(word, wordList, lists);
        }
        if (endWordIndex == -1) {
            return 0;
        }
        boolean[] visited = new boolean[wordList.size()];
        addEdge(beginWord, wordList, lists);
        Queue<Integer> queue = new LinkedList<>(lists.get(wordList.size()));
        Queue<Integer> queue2 = new LinkedList<>();
        queue2.add(endWordIndex);
        while (!queue.isEmpty() && !queue2.isEmpty()) {
            ans++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Integer index = queue.poll();
                for (Integer index2 : queue2) {
                    if (index.equals(index2)) {
                        return ans;
                    }
                }
                visited[index] = true;
                for (Integer index2 : lists.get(index)) {
                    if (!visited[index2]) {
                        queue.add(index2);
                    }
                }
            }

            ans++;
            size = queue2.size();
            for (int i = 0; i < size; i++) {
                Integer index = queue2.poll();
                for (Integer index2 : queue) {
                    if (index.equals(index2)) {
                        return ans;
                    }
                }
                visited[index] = true;
                for (Integer index2 : lists.get(index)) {
                    if (!visited[index2]) {
                        queue2.add(index2);
                    }
                }
            }
        }
        return 0;
    }

    private void addEdge(String s, List<String> wordList, List<List<Integer>> lists) {
        List<Integer> list = new ArrayList<>();
        char[] chars = s.toCharArray();
        for (int i = 0; i < wordList.size(); i++) {
            int count = 0;
            String word = wordList.get(i);
            for (int j = 0; j < word.length(); j++) {
                if (chars[j] != word.charAt(j)) {
                    count++;
                }
            }
            if (count == 1) {
                list.add(i);
            }
        }
        lists.add(list);
    }

    /**
     * 129. 求根到叶子节点数字之和
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        this.treeNodeList.add(root.val);
        if (root.left == null && root.right == null) {
            int a = 1;
            for (int i = this.treeNodeList.size() - 1; i >= 0; i--) {
                this.ans += this.treeNodeList.get(i) * a;
                a *= 10;
            }
        } else {
            sumNumbers(root.left);
            sumNumbers(root.right);
        }
        this.treeNodeList.remove(this.treeNodeList.size() - 1);
        return this.ans;
    }

    private final List<Integer> treeNodeList = new ArrayList<>();

    private int ans = 0;

    /**
     * 140. 单词拆分 II
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreak(String s, List<String> wordDict) {
        return null;
//        Map<Character, List<Integer>> map = new HashMap<>();
//        List<String> ans = new ArrayList<>();
//        Map<Integer, List<List<Integer>>> memoryMap = new HashMap<>();
//        // 初始化map
//        for (int i = 0; i < wordDict.size(); i++) {
//            char c = wordDict.get(i).charAt(0);
//            List<Integer> list = map.get(c);
//            if (list == null) {
//                list = new ArrayList<>();
//            }
//            list.add(i);
//            map.put(c, list);
//        }
//        List<List<Integer>> splitList = wordBreak(s, wordDict, map, new ArrayList<>(), 0, memoryMap);
//        // 构建结果
//        for (List<Integer> splitIndex : splitList) {
//            StringBuilder sb = new StringBuilder();
//            int pre = 0;
//            for (int i = 0; i < splitIndex.size(); i++) {
//                Integer split = splitIndex.get(i);
//                sb.append(s, pre, split).append(" ");
//                pre = split;
//            }
//            String a = sb.toString();
//            ans.add(a.substring(0, a.length() - 1));
//        }
//        return ans;
    }

//    private List<List<Integer>> wordBreak(String s, List<String> wordDict, Map<Character, List<Integer>> map,
//                           List<Integer> splitIndex, int i, Map<Integer, List<List<Integer>>> memoryMap) {
//        List<List<Integer>> res = new ArrayList<>();
//        if (i == s.length()) {
//            res.add(new ArrayList<>(splitIndex));
//            return res;
//        }
//        // 读取缓存
//        if (memoryMap.get(i) != null) {
//            return memoryMap.get(i);
//        }
//        List<Integer> list = map.get(s.charAt(i));
//        if (list == null) {
//            return res;
//        }
//        for (Integer index : list) {
//            String word = wordDict.get(index);
//            if (word.length() <= s.length() - i && word.equals(s.substring(i, i + word.length()))) {
//                splitIndex.add(i + word.length());
//                List<List<Integer>> r = wordBreak(s, wordDict, map, splitIndex, i + word.length(), memoryMap);
//                res.addAll(new ArrayList<>(r));
//                splitIndex.remove(splitIndex.size() - 1);
//            }
//        }
//        memoryMap.put(i, res);
//        return res;
//    }

    /**
     * 141. 环形链表
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode p = head;
        ListNode q = head.next;
        while (p != null && q != null && q.next != null) {
            if (p == q) {
                return true;
            }
            p = p.next;
            q = q.next.next;
        }
        return false;
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
     * 143. 重排链表
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }
        // 1.找链表中点
        ListNode p = head;
        ListNode q = head;
        while (p != null && p.next != null) {
            p = p.next.next;
            q = q.next;
        }
        p = head;
        // 2.反转后半部的链表
        q = reverseNodeList(q);
        // 3.合并两个链表
        while (p != null) {
            if (q == null) {
                p.next = null;
                break;
            }
            ListNode p2 = p.next;
            ListNode q2 = q.next;
            p.next = q;
            q.next = p2;
            p = p2;
            q = q2;
        }
    }

    private ListNode reverseNodeList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p = reverseNodeList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    /**
     * 144. 二叉树的前序遍历
     * @param root
     * @return
     */
//    public List<Integer> preorderTraversal(TreeNode root) {
//        List<Integer> ans = new ArrayList<>();
//        if (root == null) {
//            return ans;
//        }
//        Stack<TreeNode> stack = new Stack<>();
//        TreeNode node = root;
//        while (!stack.isEmpty() || node != null) {
//            while (node != null) {
//                ans.add(node.val);
//                stack.push(node);
//                node = node.left;
//            }
//            node = stack.pop();
//            node = node.right;
//        }
//        return ans;
//    }

    /**
     * 144. 二叉树的前序遍历
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }
        TreeNode p1 = root;
        TreeNode p2;
        while (p1 != null) {
            p2 = p1.left;
            if (p2 != null) {
                while (p2.right != null && p2.right != p1) {
                    p2 = p2.right;
                }
                if (p2.right == null) {
                    ans.add(p1.val);
                    p2.right = p1;
                    p1 = p1.left;
                    continue;
                } else {
                    p2.right = null;
                }
            } else {
                ans.add(p1.val);
            }
            p1 = p1.right;
        }
        return ans;
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
     * 201. 数字范围按位与
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        // 位移，寻找最大公共前缀
        int shift = 0;
        while (m < n) {
            m >>= 1;
            n >>= 1;
            ++shift;
        }
        return m << shift;
    }

    /**
     * 214. 最短回文串
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
//        StringBuilder sb = new StringBuilder(s);
//        int i = 0;
//        int j = s.length() - 1;
//        // 限制只能在原字符串左侧加，即<=limit
//        int limit = 0;
//        while (i < j) {
//            if (sb.charAt(i) != sb.charAt(j)) {
//                if (i > limit) {
//                    j += i - limit;
//                    i = limit;
//                }
//                sb.insert(i, sb.charAt(j));
//                j++;
//                limit++;
//            }
//            i++;
//            j--;
//        }
//        return sb.toString();
        // 使用KMP算法的求next数组方法
        int[] next = new int[s.length()];
        int j = 0;
        for (int i = 1; i < s.length(); i++) {
            while (j != 0 && s.charAt(j) != s.charAt(i)) {
                // 从next[i+1]的求解回溯到next[j]
                j = next[j - 1];
            }
            if (s.charAt(j) == s.charAt(i)) {
                j++;
            }
            next[i] = j;
        }

        String str = new StringBuilder(s).reverse().toString();
        int i1 = 0;
        while (i1 < str.length()) {
            int i2 = i1;
            int k = 0;
            while (i2 < str.length() && k < s.length()) {
                if (str.charAt(i2) != s.charAt(k)) {
                    i1 += next[k];
                    break;
                }
                i2++;
                k++;
            }
            if (i2 == str.length()) {
                break;
            }
            i1++;
        }
        return str.substring(0, i1) + s;
    }

    /**
     * 216. 组合总和 III
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 1; i <= 9; i++) {
            dfs(ans, new ArrayList<>(), k, n, i, 0);
        }
        return ans;
    }

    private void dfs(List<List<Integer>> ans, List<Integer> list, int k, int n, int i, int sum) {
        if (list.size() >= k) {
            return;
        }
        list.add(i);
        sum += i;
        if (sum == n && list.size() == k) {
            ans.add(new ArrayList<>(list));
        }
        if (sum < n) {
            for (i = i + 1; i <= 9; i++) {
                dfs(ans, list, k, n, i, sum);
            }
        }
        list.remove(list.size() - 1);
    }

    /**
     * 226. 翻转二叉树
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.right = left;
        root.left = right;
        return root;
    }

    /**
     * 234. 回文链表
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode p = head;
        ListNode q = head;
        while (p != null && q != null && q.next != null) {
            p = p.next;
            q = q.next.next;
        }
        if (q != null) {
            // 链表数为奇数
            q = nodeReverse(p.next);
        } else {
            // 链表数为偶数
            q = nodeReverse(p);
        }
        p = head;
        while (p != null && q != null) {
            if (p.val != q.val) {
                return false;
            }
            p = p.next;
            q = q.next;
        }
        return true;
    }

    private ListNode nodeReverse(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = head;
        ListNode current = head.next;
        ListNode next = head.next.next;
        head.next = null;
        while (current != null) {
            current.next = pre;
            pre = current;
            current = next;
            if (next != null) {
                next = next.next;
            }
        }
        return pre;
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
     * 257. 二叉树的所有路径
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        StringBuilder sb = new StringBuilder(root.val + "");
        if (root.left == null && root.right == null) {
            result.add(sb.toString());
            return result;
        }
        dfs(root.left, sb, result);
        dfs(root.right, sb, result);
        return result;
    }

    private void dfs(TreeNode root, StringBuilder sb, List<String> result) {
        if (root == null) {
            return;
        }
        sb.append("->").append(root.val);
        if (root.left == null && root.right == null) {
            result.add(sb.toString());
        }
        dfs(root.left, sb, result);
        dfs(root.right, sb, result);
        sb.delete(sb.lastIndexOf("->"), sb.length());
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
     * 332. 重新安排行程
     * @param tickets
     * @return
     */
    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, PriorityQueue<String>> map = new HashMap<>();
        List<String> ans = new LinkedList<>();
        for (List<String> ticket : tickets) {
            String from = ticket.get(0);
            String to = ticket.get(1);
            if (!map.containsKey(from)) {
                map.put(from, new PriorityQueue<>());
            }
            map.get(from).add(to);
        }
        dfs("JFK", map, ans);
        Collections.reverse(ans);
        return ans;
    }

    private void dfs(String curr, Map<String, PriorityQueue<String>> map, List<String> ans) {
        while (map.containsKey(curr) && map.get(curr).size() > 0) {
            String temp = map.get(curr).poll();
            dfs(temp, map, ans);
        }
        ans.add(curr);
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
     * 344. 反转字符串
     * @param s
     */
    public void reverseString(char[] s) {
        int i = 0;
        int j = s.length - 1;
        while (i < j) {
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }

    /**
     * 347. 前 K 个高频元素
     * @param nums
     * @param k
     * @return
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (!map.containsKey(num)) {
                map.put(num, 0);
            }
            map.put(num, map.get(num) + 1);
        }
        int[][] array = new int[nums.length + 1][];
        for (int key : map.keySet()) {
            int value = map.get(key);
            if (array[value] == null) {
                array[value] = new int[]{key};
            } else {
                int[] tempArray = new int[array[value].length + 1];
                System.arraycopy(array[value], 0, tempArray, 0, array[value].length);
                tempArray[array[value].length] = key;
                array[value] = tempArray;
            }
        }
        int[] ans = new int[k];
        int count = 0;
        for (int i = array.length - 1; i >= 0; i--) {
            if (array[i] == null) {
                continue;
            }
            for (int j = 0; j < array[i].length; j++) {
                ans[count++] = array[i][j];
                if (count >= k) {
                    return ans;
                }
            }
        }
        return ans;
    }

    /**
     * 349. 两个数组的交集
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        for (int i1 : nums1) {
            set1.add(i1);
        }
        for (int i2 : nums2) {
            set2.add(i2);
        }
        set1.retainAll(set2);
        int[] res = new int[set1.size()];
        int i = 0;
        for (int a : set1) {
            res[i] = a;
            i++;
        }
        return res;
    }

    /**
     * 402. 移掉K位数字
     * @param num
     * @param k
     * @return
     */
    public String removeKdigits(String num, int k) {
        if (k == 0) {
            return String.valueOf(Integer.parseInt(num));
        }
        if (num.length() == k) {
            return "0";
        }
        num = removeKdigits1(num, k);
        int i = 0;
        while (i < num.length() && num.charAt(i) == '0') {
            i++;
        }
        String ans = num.substring(i);
        if (ans.length() == 0) {
            return "0";
        }
        return ans;
    }

    private String removeKdigits1(String num, int k) {
        if (k == 0) {
            return String.valueOf(Integer.parseInt(num));
        }
        if (num.length() == k) {
            return "";
        }
        int minValue = Integer.MAX_VALUE;
        for (int i = 0; i < num.length(); i++) {
            minValue = Math.min(Integer.parseInt(num.substring(i, i + 1)), minValue);
        }
        int minIndex = num.indexOf(String.valueOf(minValue));
        String ans;
        if (minIndex < k) {
            ans = minValue + removeKdigits1(num.substring(minIndex + 1), k - minIndex);
        } else if (minIndex == k) {
            ans = num.substring(minIndex);
        } else {
            ans = removeKdigits1(num.substring(0, minIndex), k) + num.substring(minIndex);
        }
        return ans;
    }

    /**
     * 404. 左叶子之和
     * @param root
     * @return
     */
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return 0;
        }
        int sum = 0;
        if (root.left != null && root.left.left == null && root.left.right == null) {
            sum += root.left.val;
        }
        return sum + sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
    }

    /**
     * 406. 根据身高重建队列
     * @param people
     * @return
     */
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (person1, person2) -> {
            if (person1[0] == person2[0]) {
                return person1[1] - person2[1];
            } else {
                return person2[0] - person1[0];
            }
        });
        List<int[]> ans = new ArrayList<>();
        for (int[] person : people) {
            ans.add(person[1], person);
        }
        return ans.toArray(new int[people.length][]);
    }

    /**
     * 416. 分割等和子集
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        if (nums.length == 0) {
            return false;
        }
        Arrays.sort(nums);
        int target = 0;
        for (int num : nums) {
            target += num;
        }
        if (target % 2 == 1) {
            return false;
        }
        target /= 2;
        if (nums[nums.length - 1] > target) {
            return false;
        }
        boolean[][] dp = new boolean[nums.length][target + 1];
        dp[0][0] = true;
        dp[0][nums[0]] = true;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j <= target; j++) {
                if (dp[i - 1][j]) {
                    dp[i][j] = true;
                    if (j + nums[i] <= target) {
                        dp[i][j + nums[i]] = true;
                    }
                }
            }
        }
        return dp[nums.length - 1][target];
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
     * 459. 重复的子字符串
     * @param s
     * @return
     */
    public boolean repeatedSubstringPattern(String s) {
//        if (s.length() == 1) {
//            return false;
//        }
//        char endChar = s.charAt(s.length() - 1);
//        for (int i = 0; i < s.length() - 1; ++i) {
//            if (s.length() % (i + 1) == 0 && s.charAt(i) == endChar) {
//                String substring = s.substring(0, i + 1);
//                boolean flag = true;
//                for (int j = 1; j < s.length() / (i + 1); ++j) {
//                    if (!substring.equals(s.substring(j * (i + 1), j * (i + 1) + i + 1))) {
//                        flag = false;
//                        break;
//                    }
//                }
//                if (flag) {
//                    return true;
//                }
//            }
//        }
//        return false;
        // 双倍字符串法
        return (s + s).indexOf(s, 1) != s.length();
    }

    /**
     * 486. 预测赢家
     * @param nums
     * @return
     */
    public boolean PredictTheWinner(int[] nums) {
        // 动态规划
        int[] dp = new int[nums.length];
        System.arraycopy(nums, 0, dp, 0, nums.length);
        for (int i = nums.length - 1; i >= 0; i--) {
            for (int j = i + 1; j < nums.length; j++) {
                dp[j] = Math.max(nums[i] - dp[j], nums[j] - dp[j - 1]);
            }
        }
        return dp[nums.length - 1] >= 0;
    }

    /**
     * 491. 递增子序列
     * @param nums
     * @return
     */
    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        dfs(ans, temp, 0, Integer.MIN_VALUE, nums);
        return ans;
    }

    private void dfs(List<List<Integer>> ans, List<Integer> temp, int cur, int last, int[] nums) {
        if (cur == nums.length) {
            if (temp.size() >= 2) {
                ans.add(new ArrayList<>(temp));
            }
            return;
        }
        if (nums[cur] >= last) {
            temp.add(nums[cur]);
            dfs(ans, temp, cur + 1, nums[cur], nums);
            temp.remove(temp.size() - 1);
        }
        if (nums[cur] != last) {
            dfs(ans, temp, cur + 1, last, nums);
        }
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
     * 530. 二叉搜索树的最小绝对差
     * @param root
     * @return
     */
    public int getMinimumDifference(TreeNode root) {
        dfs(root);
        return min;
    }

    private Integer pre = null;

    private Integer min = Integer.MAX_VALUE;

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        if (pre != null) {
            min = Math.min(min, Math.abs(root.val - pre));
        }
        pre = root.val;
        dfs(root.right);
    }

    /**
     * 538. 把二叉搜索树转换为累加树
     * @param root
     * @return
     */
    public TreeNode convertBST(TreeNode root) {
        if (root != null) {
            convertBST(root.right);
            sum += root.val;
            root.val = sum;
            convertBST(root.left);
        }
        return root;
    }

    private int sum = 0;

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
     * 557. 反转字符串中的单词 III
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        String[] strs = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (String str : strs) {
            sb.append(" ").append(new StringBuilder(str).reverse());
        }
        return sb.toString().substring(1);
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
     * 657. 机器人能否返回原点
     * @param moves
     * @return
     */
    public boolean judgeCircle(String moves) {
        // 水平方向
        int horizontal = 0;
        // 垂直方向
        int vertical = 0;
        for (int i = 0; i < moves.length(); ++i) {
            char c = moves.charAt(i);
            if (c == 'R') {
                ++horizontal;
            } else if (c == 'L') {
                --horizontal;
            } else if (c == 'U') {
                ++vertical;
            } else {
                --vertical;
            }
        }
        return horizontal == 0 && vertical == 0;
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
     * 834. 树中距离之和
     * @param N
     * @param edges
     * @return
     */
    public int[] sumOfDistancesInTree(int N, int[][] edges) {
        int[] ans = new int[N];
        Map<Integer, List<Integer>> map = new HashMap<>(N);
        // 初始化各节点连接的节点数组映射map
        for (int i = 0; i < N; i++) {
            map.put(i, new ArrayList<>());
        }
        for (int[] edge : edges) {
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }
        // 遍历计算各个节点
        for (int i = 0; i < N; i++) {
            Queue<Integer> queue = new LinkedList<>();
            queue.add(i);
            int modulus = 1;
            boolean[] isVisit = new boolean[N];
            while (!queue.isEmpty()) {
                int size = queue.size();
                for (int j = 0; j < size; j++) {
                    Integer node = queue.remove();
                    List<Integer> list = map.get(node);
                    isVisit[node] = true;
                    List<Integer> newList = new ArrayList<>();
                    // 去除已经遍历过的
                    for (Integer temp : list) {
                        if (!isVisit[temp]) {
                            newList.add(temp);
                        }
                    }
                    ans[i] += newList.size() * modulus;
                    queue.addAll(newList);
                }
                modulus++;
            }
        }
        return ans;
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
     * 841. 钥匙和房间
     * @param rooms
     * @return
     */
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int[] flag = new int[rooms.size()];
        flag[0] = 1;
        dfs(rooms, flag, 0);
        int count = 0;
        for (int f : flag) {
            count += f;
        }
        return count == rooms.size();
    }

    private void dfs(List<List<Integer>> rooms, int[] flag, int i) {
        if (flag[i] == 1) {
            List<Integer> room = rooms.get(i);
            for (Integer r : room) {
                if (flag[r] == 0) {
                    flag[r] = 1;
                    dfs(rooms, flag, r);
                }
            }
        }
    }

    /**
     * 844. 比较含退格的字符串
     * @param S
     * @param T
     * @return
     */
    public boolean backspaceCompare(String S, String T) {
        int skip1 = 0;
        int skip2 = 0;
        int i = S.length() - 1;
        int j = T.length() - 1;
        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (S.charAt(i) == '#') {
                    skip1++;
                } else if (skip1 > 0){
                    skip1--;
                } else {
                    break;
                }
                i--;
            }
            while (j >= 0) {
                if (T.charAt(j) == '#') {
                    skip2++;
                } else if (skip2 > 0) {
                    skip2--;
                } else {
                    break;
                }
                j--;
            }
            if (i >= 0 && j >= 0) {
                if (S.charAt(i) != T.charAt(j)) {
                    return false;
                }
            } else if (i >= 0 || j >= 0) {
                return false;
            }
            i--;
            j--;
        }
        return true;
    }

    /**
     * 845. 数组中的最长山脉
     * @param A
     * @return
     */
    public int longestMountain(int[] A) {
        int ans = 0;
        int count = 1;
        int i = 1;
        int status = 0;
        while (i < A.length) {
            if (status == 0 && A[i - 1] < A[i]) {
                status = 1;
                count++;
            } else if (status == 1) {
                if (A[i - 1] == A[i]) {
                    status = 0;
                    count = 0;
                }
                if (A[i - 1] > A[i]) {
                    status = 2;
                }
                count++;
            } else if (status == 2) {
                if (A[i - 1] <= A[i]) {
                    ans = Math.max(ans, count);
                    if (A[i - 1] < A[i]) {
                        status = 1;
                        count = 1;
                    } else {
                        status = 0;
                        count = 0;
                    }
                }
                count++;
            }
            i++;
        }
        if (status == 2) {
            ans = Math.max(ans, count);
        }
        return ans;
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
     * 922. 按奇偶排序数组 II
     * @param A
     * @return
     */
    public int[] sortArrayByParityII(int[] A) {
        // 偶数
        int i = 0;
        // 奇数
        int j = 1;
        int[] res = new int[A.length];
        for (int value : A) {
            if (value % 2 == 0) {
                res[i] = value;
                i += 2;
            } else {
                res[j] = value;
                j += 2;
            }
        }
        return res;
    }

    /**
     * 925. 长按键入
     * @param name
     * @param typed
     * @return
     */
    public boolean isLongPressedName(String name, String typed) {
        if (name.length() == 0) {
            return false;
        }
        int i = 0;
        int j = 0;
        char c = name.charAt(0);
        while (i < name.length() && j < typed.length()) {
            if (name.charAt(i) == typed.charAt(j)) {
                c = name.charAt(i);
                i++;
                j++;
            } else {
                if (typed.charAt(j) == c) {
                    j++;
                } else {
                    return false;
                }
            }
        }
        while (j < typed.length()) {
            if (typed.charAt(j) != c) {
                return false;
            }
            j++;
        }
        return i == name.length();
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
     * 941. 有效的山脉数组
     * @param A
     * @return
     */
    public boolean validMountainArray(int[] A) {
        if (A.length < 3) {
            return false;
        }
        int i = 1;
        while (i < A.length && A[i - 1] < A[i]) {
            i++;
        }
        if (i == 1 || i == A.length) {
            return false;
        }
        while (i < A.length && A[i - 1] > A[i]) {
            i++;
        }
        return i == A.length;
    }

    /**
     * 973. 最接近原点的 K 个点
     * @param points
     * @param K
     * @return
     */
    public int[][] kClosest(int[][] points, int K) {
        fastSelect(points, K, 0, points.length - 1);
        int[][] res = new int[K][];
        if (K >= 0) System.arraycopy(points, 0, res, 0, K);
        return res;
    }

    private void fastSelect(int[][] points, int K, int left, int right) {
        if (points.length == 0 || left >= right) {
            return;
        }
        int i = left;
        int j = right;
        int a = getDistance(points, i);
        while (i < j) {
            while (i < j && getDistance(points, j) >= a) {
                j--;
            }
            swap(points, i, j);
            while (i < j && getDistance(points, i) <= a) {
                i++;
            }
            swap(points, i, j);
        }
        if (i == K) {
            return;
        }
        if (i < K) {
            fastSelect(points, K, i + 1, right);
        } else {
            fastSelect(points, K, left, i - 1);
        }
    }

    private int getDistance(int[][] points, int index) {
        return points[index][0] * points[index][0] + points[index][1] * points[index][1];
    }

    private void swap(int[][] points, int i, int j) {
        int[] temp = points[i];
        points[i] = points[j];
        points[j] = temp;
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
     * 1002. 查找常用字符
     * @param A
     * @return
     */
    public List<String> commonChars(String[] A) {
        if (A.length == 0) {
            return new ArrayList<>();
        }
        int[][] count = new int[A.length][26];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length(); j++) {
                count[i][A[i].charAt(j) - 'a'] += 1;
            }
        }
        List<String> ans = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            int n = Integer.MAX_VALUE;
            for (int j = 0; j < A.length; j++) {
                n = Math.min(n, count[j][i]);
            }
            for (int j = 0; j < n; j++) {
                ans.add(String.valueOf((char)(i + 'a')));
            }
        }
        return ans;
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
        return sum1 % mod;
    }

    private void dfs(TreeNode root, int val) {
        if (root == null) {
            return;
        }
        val = val << 1 | root.val;
        if (root.left == null && root.right == null) {
            sum1 += val;
        }
        dfs(root.left, val);
        dfs(root.right, val);
    }

    private int sum1;

    /**
     * 1024. 视频拼接
     * @param clips
     * @param T
     * @return
     */
    public int videoStitching(int[][] clips, int T) {
        int[] greed = new int[T];
        for (int[] clip : clips) {
            if (clip[0] < T) {
                greed[clip[0]] = Math.max(greed[clip[0]], clip[1]);
            }

        }
        int pre = 0;
        int last = 0;
        int ans = 0;
        for (int i = 0; i < T; i++) {
            last = Math.max(last, greed[i]);
            if (last == i) {
                return -1;
            }
            if (pre == i) {
                ans++;
                pre = last;
            }
        }
        return ans;
    }

    /**
     * 1030. 距离顺序排列矩阵单元格
     * @param R
     * @param C
     * @param r0
     * @param c0
     * @return
     */
    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        int[][] res = new int[R * C][];
        boolean[][] visit = new boolean[R][C];
        int k = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{r0, c0});
        visit[r0][c0] = true;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] node = queue.poll();
                res[k++] = node;
                if (node[0] - 1 >= 0 && !visit[node[0] - 1][node[1]]) {
                    visit[node[0] - 1][node[1]] = true;
                    queue.add(new int[]{node[0] - 1, node[1]});
                }
                if (node[0] + 1 < R && !visit[node[0] + 1][node[1]]) {
                    visit[node[0] + 1][node[1]] = true;
                    queue.add(new int[]{node[0] + 1, node[1]});
                }
                if (node[1] - 1 >= 0 && !visit[node[0]][node[1] - 1]) {
                    visit[node[0]][node[1] - 1] = true;
                    queue.add(new int[]{node[0], node[1] - 1});
                }
                if (node[1] + 1 < C && !visit[node[0]][node[1] + 1]) {
                    visit[node[0]][node[1] + 1] = true;
                    queue.add(new int[]{node[0], node[1] + 1});
                }
            }
        }
        return res;
    }

    /**
     * 1122. 数组的相对排序
     * @param arr1
     * @param arr2
     * @return
     */
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int arr : arr1) {
            countMap.put(arr, countMap.getOrDefault(arr, 0) + 1);
        }
        int k = 0;
        for (int arr : arr2) {
            int count = countMap.get(arr);
            for (int i = 0; i < count; i++) {
                arr1[k++] = arr;
            }
            countMap.remove(arr);
        }
        int splitIndex = k;
        for (int key : countMap.keySet()) {
            int count = countMap.get(key);
            for (int i = 0; i < count; i++) {
                arr1[k++] = key;
            }
        }
        Arrays.sort(arr1, splitIndex, arr1.length);
        return arr1;
    }

    /**
     * 1207. 独一无二的出现次数
     * @param arr
     * @return
     */
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int a : arr) {
            map.put(a, map.getOrDefault(a, 0) + 1);
        }
        Map<Integer, Integer> times = new HashMap<>();
        for (int key : map.keySet()) {
            int time = map.get(key);
            if (times.containsKey(time)) {
                return false;
            }
            times.put(time, 1);
        }
        return true;
    }

    /**
     * 1356. 根据数字二进制下 1 的数目排序
     * @param arr
     * @return
     */
    public int[] sortByBits(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = Integer.bitCount(arr[i]) * 100000 + arr[i];
        }
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; i++) {
            arr[i] %= 100000;
        }
        return arr;
    }

    /**
     * 1365. 有多少小于当前数字的数字
     * @param nums
     * @return
     */
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] count = new int[101];
        for (int num : nums) {
            count[num]++;
        }
        int[] ans = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            int c = 0;
            for (int j = 0; j < nums[i]; j++) {
                c += count[j];
            }
            ans[i] = c;
        }
        return ans;
    }
}
