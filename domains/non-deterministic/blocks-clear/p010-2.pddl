(define (problem blocks-010-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b3) (clear b5) (clear b6) (handempty) (on b2 b7) (on b4 b8) (on b5 b4) (on b7 b0) (on b8 b9) (on b9 b2) (ontable b0) (ontable b1) (ontable b3) (ontable b6))
    (:goal (clear b4))
)