(define (problem BLOCKS-10-1)
(:domain BLOCKS)
(:objects D A J I E G H B F C )
(:init (clear C) (clear F) (ontable B) (ontable H) (on C G) (on G E) (on E I)
 (on I J) (on J A) (on A B) (on F D) (on D H) (handempty))
(:goal (and (on C B) (on B D) (on D F) (on F I) (on I A) (on A E) (on E H)
            (on H G) (on G J)))
)