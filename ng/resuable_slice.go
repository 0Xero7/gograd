package ng

type TensorChildren struct {
	children []*Tensor
	capacity int
	ptr      int
}

func NewTensorChildren() *TensorChildren {
	r := new(TensorChildren)
	r.children = make([]*Tensor, 0)
	r.capacity = 0
	r.ptr = 0
	return r
}

func NewTensorChildrenWith(children []*Tensor) *TensorChildren {
	r := new(TensorChildren)
	r.children = make([]*Tensor, len(children))
	copy(r.children, children)
	r.capacity = len(children)
	r.ptr = len(children)
	return r
}

func (c *TensorChildren) Clone() *TensorChildren {
	r := new(TensorChildren)
	r.children = make([]*Tensor, len(c.children))
	copy(r.children, c.children)
	r.ptr = c.ptr
	r.capacity = c.capacity
	return r
}

func (c *TensorChildren) At(index int) *Tensor {
	return c.children[index]
}

func (c *TensorChildren) Set(index int, value *Tensor) {
	c.children[index] = value
}

func (c *TensorChildren) Append(value *Tensor) {
	if c.ptr == c.capacity {
		newSize := c.capacity
		if newSize == 0 {
			newSize = 1
		} else {
			newSize = newSize * 2
		}
		c.capacity = newSize

		newPool := make([]*Tensor, newSize)
		copy(newPool[0:c.capacity], c.children)
		c.children = newPool
	}

	c.children[c.ptr] = value
	c.ptr++
}

func (c *TensorChildren) Len() int {
	return c.ptr
}

func (c *TensorChildren) Clear() {
	c.ptr = 0
}
