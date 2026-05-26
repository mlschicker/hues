use std::ops::BitXorAssign;

#[derive(Clone, Debug)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize, // actual bit count
}

impl BitSet {
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
            len,
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        self.words.resize(new_len.div_ceil(64), 0);
        self.len = new_len;
    }

    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.len);
        let word_index = index / 64;
        let bit_index = index % 64;
        if value {
            self.words[word_index] |= 1 << bit_index;
        } else {
            self.words[word_index] &= !(1 << bit_index);
        }
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        let len = self.len;
        let n_words = len.div_ceil(64);
        self.words[..n_words].iter().copied().enumerate().flat_map(move |(word_idx, word)| {
            let start = word_idx * 64;
            let mut w = if start + 64 > len {
                word & ((1u64 << (len - start)) - 1)
            } else {
                word
            };
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(start + bit)
            })
        })
    }

    #[inline]
    pub fn xor_assign(&mut self, other: &BitSet) {
        // LLVM will auto-vectorize this
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a ^= b;
        }
    }

    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        let word_index = index / 64;
        let bit_index = index % 64;
        self.words[word_index] & (1 << bit_index) != 0
    }
}

impl BitXorAssign<&BitSet> for BitSet {
    fn bitxor_assign(&mut self, rhs: &BitSet) {
        self.xor_assign(rhs);
    }
}

