package utils;

import common.ListNode;

import java.util.Scanner;

public class ListNodeUtil {

    public static ListNode generateListNode() {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入链表元素，以空格分割");
        String line = sc.nextLine();
        String[] split = line.split("\\s+");
        ListNode tempNode = new ListNode(0);
        ListNode p = tempNode;
        for (String s : split) {
            int i = Integer.parseInt(s);
            ListNode node = new ListNode(i);
            p.next = node;
            p = node;
        }
        return tempNode.next;
    }
}
