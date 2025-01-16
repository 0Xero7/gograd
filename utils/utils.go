package utils

func AssertTrue(condition bool, errorMessage string) {
	if !condition {
		panic(errorMessage)
	}
}
