(define (problem BLOCKS-5-CLEAR)
(:domain BLOCKS)
(:objects A B C D E)
(:init
    (ontable A)
    (ontable B)
    (ontable C)
    (ontable D)
    (ontable E)
    (clear A)
    (clear B)
    (clear C)
    (clear D)
    (clear E)
    (handempty))
(:goal (and (clear A))))
