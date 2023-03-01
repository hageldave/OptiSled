package hageldave.utils;

import java.util.LinkedList;
import java.util.function.BiConsumer;

/** Pointer/Reference class. With notification ability */
public final class Ref<T> {

	public T r;
	
	private LinkedList<BiConsumer<T,T>> listeners;
	
	public Ref(T r) {
		this.r=r;
	}
	
	public Ref() {}
	
	public T get() {
		return r;
	}
	
	public void set(T r) {
		T prev = this.r;
		this.r = r;
		notifyListeners(prev, r);
	}
	
	public boolean isNull() {
		return r == null;
	}
	
	public static <T> Ref<T> of(T r){
		return new Ref<>(r);
	}
	
	public synchronized BiConsumer<T,T> addListener(BiConsumer<T,T> l){
		if(listeners == null)
			listeners = new LinkedList<>();
		listeners.add(l);
		return l;
	}
	
	public synchronized void removeListener(BiConsumer<T,T> l) {
		if(listeners != null)
			listeners.remove(l);
	}
	
	private synchronized void notifyListeners(T prev,T curr) {
		if(listeners != null)
			listeners.forEach(l->l.accept(prev,curr));
	}
}
