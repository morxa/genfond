(define (problem blocks-009-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b2) (clear b4) (clear b6) (clear b8) (handempty) (on b0 b3) (on b1 b5) (on b2 b1) (on b7 b0) (on b8 b7) (ontable b3) (ontable b4) (ontable b5) (ontable b6))
    (:goal (on b6 b3))
)