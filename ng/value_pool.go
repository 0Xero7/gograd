package ng

type ValuePool[T any] struct {
	pool     []*T
	capacity int
	ptr      int
	mark     int

	factory func(index int) *T
}

func NewValuePool[T any](factory func(index int) *T) ValuePool[T] {
	pool := ValuePool[T]{}
	pool.pool = make([]*T, 0)
	pool.ptr = 0
	pool.mark = 0
	pool.capacity = 0
	pool.factory = factory
	return pool
}

func (v *ValuePool[T]) Get() *T {
	if v.ptr == v.capacity {
		newSize := v.capacity
		if newSize == 0 {
			newSize = 1
		} else {
			newSize = newSize * 2
		}
		v.capacity = newSize

		newPool := make([]*T, newSize)
		copy(newPool[0:v.capacity], v.pool)
		v.pool = newPool
	}

	obj := v.pool[v.ptr]
	if obj == nil {
		obj = v.factory(v.ptr)
		v.pool[v.ptr] = obj
	}
	v.ptr++
	return obj
}

func (v *ValuePool[T]) Reset() {
	v.ptr = v.mark
}

func (v *ValuePool[T]) Mark() {
	v.mark = v.ptr
}

func (v *ValuePool[T]) GetCapacity() int {
	return v.capacity
}
