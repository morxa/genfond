(define (problem blocks-011-03)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b3) (clear b6) (clear b7) (clear b9) (handempty) (on b0 b2) (on b10 b5) (on b3 b8) (on b4 b10) (on b5 b1) (on b6 b0) (on b8 b4) (ontable b1) (ontable b2) (ontable b7) (ontable b9))
    (:goal (on b9 b2))
)