(define (problem BLOCKS-5-2)
(:domain BLOCKS)
(:objects A C E B D )
(:init (clear D) (ontable B) (on D E) (on E C) (on C A) (on A B) (handempty))
(:goal (and (on D C) (on C B) (on B E) (on E A)))
)