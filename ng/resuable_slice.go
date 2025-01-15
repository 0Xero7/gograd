package ng

type Children struct {
	children []*Value
	capacity int
	ptr      int
}

func NewChildren() *Children {
	r := new(Children)
	r.children = make([]*Value, 0)
	r.capacity = 0
	r.ptr = 0
	return r
}

func (c *Children) At(index int) *Value {
	return c.children[index]
}

func (c *Children) Set(index int, value *Value) {
	c.children[index] = value
}

func (c *Children) Append(value *Value) {
	if c.ptr == c.capacity {
		newSize := c.capacity
		if newSize == 0 {
			newSize = 1
		} else {
			newSize = newSize * 2
		}
		c.capacity = newSize

		newPool := make([]*Value, newSize)
		copy(newPool[0:c.capacity], c.children)
		c.children = newPool
	}

	c.children[c.ptr] = value
	c.ptr++
}

func (c *Children) Len() int {
	return c.ptr
}

func (c *Children) Clear() {
	c.ptr = 0
}
