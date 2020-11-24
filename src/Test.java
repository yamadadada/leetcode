import common.ListNode;
import common.TreeNode;
import utils.ArrayUtil;
import utils.ListNodeUtil;
import utils.ListUtil;

import java.util.ArrayList;
import java.util.List;

public class Test {

    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        root.left = node2;
        root.right = node3;
        TreeNode node4 = new TreeNode(4);
        TreeNode node5 = new TreeNode(5);
        node2.left = node4;
        node2.right = node5;
        TreeNode node6 = new TreeNode(6);
        node3.left = node6;
        Solution solution = new Solution();
        solution.countNodes(root);
    }
}
