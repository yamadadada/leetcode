import common.TreeNode;
import utils.ArrayUtil;
import utils.ListUtil;

import java.util.List;

public class Test {

    public static void main(String[] args) {
        Solution solution = new Solution();
        TreeNode root = new TreeNode(1);
        TreeNode node = new TreeNode(3);
        root.right = node;
        node.left = new TreeNode(2);
        int result = solution.getMinimumDifference(root);
        System.out.println(result);
    }
}
