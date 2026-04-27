use nalgebra::{DMatrix, DVector};

#[derive(Clone)]
pub struct LowerTriangularMatrix {
    size: usize,
    diagonal: DVector<f64>,
    lower_elements: DVector<f64>,
}

impl LowerTriangularMatrix {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            diagonal: DVector::from_element(size, 1.0),
            lower_elements: DVector::zeros(size * (size - 1) / 2),
        }
    }

    pub fn new_with_values(size: usize, diagonal_value: f64, lower_value: f64) -> Self {
        Self {
            size,
            diagonal: DVector::from_element(size, diagonal_value),
            lower_elements: DVector::from_element(size * (size - 1) / 2, lower_value),
        }
    }

    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut mat = DMatrix::zeros(self.size, self.size);
        let mut idx = 0;

        for i in 0..self.size {
            mat[(i, i)] = self.diagonal[i];
            for j in 0..i {
                mat[(i, j)] = self.lower_elements[idx];
                idx += 1;
            }
        }

        return mat;
    }

    pub fn get_diagonal(&self) -> DVector<f64> {
        return self.diagonal.clone();
    }

    pub fn set_diagonal(&mut self, diagonal: DVector<f64>) {
        self.diagonal = diagonal;
    }

    pub fn get_lower_elements(&self) -> DVector<f64> {
        return self.lower_elements.clone();
    }

    pub fn set_lower_elements(&mut self, lower_elements: DVector<f64>) {
        self.lower_elements = lower_elements;
    }

    pub fn get_size(&self) -> usize {
        return self.size;
    }

    pub fn get_num_parameters(&self) -> usize {
        return self.size + self.size * (self.size - 1) / 2;
    }

    pub fn get_parameters_as_vector(&self) -> DVector<f64> {
        return DVector::from_iterator(
            self.get_num_parameters(),
            self.diagonal
                .iter()
                .cloned()
                .chain(self.lower_elements.iter().cloned()),
        );
    }

    pub fn set_parameters_from_vector(&mut self, params: &DVector<f64>) {
        let size = self.size;
        self.diagonal = params.rows(0, size).into_owned();
        self.lower_elements = params.rows(size, size * (size - 1) / 2).into_owned();
    }
}
