(define (problem BLOCKS-6-1)
(:domain BLOCKS)
(:objects F D C E B A )
(:init (clear A) (clear B) (clear E) (clear C) (clear D) (ontable F)
 (ontable B) (ontable E) (ontable C) (ontable D) (on A F) (handempty))
(:goal (and (on E F) (on F C) (on C B) (on B A) (on A D)))
)