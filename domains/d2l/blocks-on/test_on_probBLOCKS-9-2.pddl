(define (problem p-9-2)
(:domain blocks)
(:objects b i c e d a g f h )
(:init (clear h) (clear f) (ontable g) (ontable f) (on h a) (on a d) (on d e)
 (on e c) (on c i) (on i b) (on b g) (handempty))
(:goal (and (on a b)))
)