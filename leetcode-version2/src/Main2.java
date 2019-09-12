import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

public class Main2 {
    public static void main(String[] args) {
        LinkedList linkedList = new LinkedList();

    }
}

enum Singleon{
    INSTANCE;
    private Person person;
    private  Singleon(){
        person = new Person();
    }

    public Person getInstace(){
        return person;
    }
}

class Person{}

class Base{

    public static void say(){
        System.out.println("base");
    }
}

class Sub extends  Base{
    public static void say(){
        System.out.println("sub");
    }
}