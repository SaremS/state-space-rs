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


#[derive(Clone)]
#[allow(non_snake_case)]
pub struct SchurStableMatrix {
    dim: usize,
    A: DMatrix<f64>,
    B: DMatrix<f64>,
}

impl SchurStableMatrix {
    /* Transforms two square matrices into a Schur stable matrix using the Cayley transform. The resulting matrix is guaranteed to be Schur stable by construction, as the Cayley transform maps the left half-plane to the unit disk. The parameters are the entries of A and B, which can be optimized freely without constraints.
    */

    #[allow(non_snake_case)]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            A: LowerTriangularMatrix::new_with_values(dim, 0.5, 0.5).to_dense(),
            B: DMatrix::identity(dim, dim),
        }
    }

    #[allow(non_snake_case)]
    pub fn to_dense(&self) -> DMatrix<f64> {
        let S = &self.A - &self.A.transpose();
        let P = &self.B * &self.B.transpose();

        let A_C = &S - &P;
        let I = DMatrix::identity(self.dim, self.dim);
        
        (&I+&A_C) * (&I-&A_C).try_inverse().unwrap() //always invertible by construction
    }

    pub fn get_num_parameters(&self) -> usize {
        2 * self.dim * self.dim
    }

    pub fn get_parameters_as_vector(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.get_num_parameters(),
            self.A.iter().cloned().chain(self.B.iter().cloned()),
        )
    }

    pub fn set_parameters_from_vector(&mut self, params: &DVector<f64>) {
        let size = self.dim * self.dim;
        self.A = DMatrix::from_row_slice(self.dim, self.dim, (&params.rows(0, size)).into());
        self.B = DMatrix::from_row_slice(self.dim, self.dim, (&params.rows(size, size)).into());
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_triangular_matrix() {
        let size = 3;
        let mut ltm = LowerTriangularMatrix::new(size);
        ltm.set_diagonal(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        ltm.set_lower_elements(DVector::from_vec(vec![0.5, 0.5, 0.5]));
        let dense = ltm.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(1, 1)], 2.0);
        assert_eq!(dense[(2, 2)], 3.0);
        assert_eq!(dense[(1, 0)], 0.5);
        assert_eq!(dense[(2, 0)], 0.5);
        assert_eq!(dense[(2, 1)], 0.5);
    }

    #[test]
    fn test_schur_stable_matrix() {
        let dim = 2;
        let mut ssm = SchurStableMatrix::new(dim);
        let dense = ssm.to_dense();
        // Check that the eigenvalues of the resulting matrix are inside the unit circle
        let eigvals = dense.clone().eigenvalues();
        for eig in eigvals.iter() {
            assert!(eig.norm() <= 1.0);
        }

        // variation 1 - A all zeros
        ssm.set_parameters_from_vector(&DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])); 
        let dense = ssm.to_dense();
        let eigvals = dense.clone().eigenvalues();
        for eig in eigvals.iter() {
            println!("Eigenvalue: {:?}", eig);
            assert!(eig.norm() <= 1.0);
        }

        // variation 2 - B all zeros
        ssm.set_parameters_from_vector(&DVector::from_vec(vec![0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])); 
        let dense = ssm.to_dense();
        let eigvals = dense.clone().eigenvalues();
        for eig in eigvals.iter() {
            println!("Eigenvalue: {:?}", eig);
            assert!(eig.norm() <= 1.0);
        }
    }
}


