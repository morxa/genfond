(define (problem blocks-009-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b1) (clear b4) (clear b5) (clear b7) (handempty) (on b0 b2) (on b3 b0) (on b5 b8) (on b7 b6) (on b8 b3) (ontable b1) (ontable b2) (ontable b4) (ontable b6))
    (:goal (on b8 b2))
)