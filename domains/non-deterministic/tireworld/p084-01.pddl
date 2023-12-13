(define (problem tireworld-084-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l9 - location)
    (:init (road l0 l72) (road l0 l80) (road l1 l24) (road l1 l67) (road l10 l6) (road l10 l81) (road l11 l17) (road l11 l51) (road l12 l31) (road l12 l78) (road l13 l31) (road l13 l35) (road l13 l44) (road l13 l5) (road l14 l20) (road l14 l63) (road l14 l79) (road l15 l16) (road l15 l19) (road l16 l15) (road l16 l39) (road l17 l11) (road l17 l73) (road l17 l81) (road l18 l25) (road l18 l83) (road l19 l15) (road l19 l42) (road l2 l4) (road l2 l48) (road l20 l14) (road l20 l61) (road l21 l45) (road l22 l62) (road l22 l70) (road l23 l33) (road l23 l51) (road l24 l1) (road l24 l66) (road l25 l18) (road l25 l61) (road l26 l35) (road l26 l70) (road l27 l59) (road l27 l65) (road l28 l64) (road l28 l69) (road l29 l40) (road l29 l78) (road l3 l30) (road l3 l60) (road l30 l3) (road l30 l71) (road l31 l12) (road l31 l13) (road l32 l66) (road l32 l76) (road l33 l23) (road l33 l58) (road l34 l71) (road l34 l82) (road l35 l13) (road l35 l26) (road l36 l50) (road l36 l79) (road l37 l53) (road l37 l75) (road l38 l43) (road l38 l54) (road l39 l16) (road l39 l48) (road l4 l2) (road l4 l47) (road l40 l29) (road l40 l75) (road l41 l5) (road l41 l8) (road l42 l19) (road l42 l74) (road l43 l38) (road l43 l57) (road l44 l13) (road l44 l69) (road l45 l21) (road l45 l59) (road l46 l77) (road l47 l4) (road l47 l64) (road l48 l2) (road l48 l39) (road l49 l58) (road l49 l65) (road l5 l13) (road l5 l41) (road l50 l36) (road l50 l68) (road l51 l11) (road l51 l23) (road l52 l54) (road l52 l60) (road l53 l37) (road l53 l82) (road l54 l38) (road l54 l52) (road l55 l77) (road l55 l82) (road l56 l73) (road l56 l76) (road l57 l43) (road l57 l68) (road l58 l33) (road l58 l49) (road l59 l27) (road l59 l45) (road l6 l10) (road l6 l8) (road l60 l3) (road l60 l52) (road l61 l20) (road l61 l25) (road l62 l22) (road l62 l74) (road l63 l14) (road l63 l67) (road l64 l28) (road l64 l47) (road l65 l27) (road l65 l49) (road l66 l24) (road l66 l32) (road l67 l1) (road l67 l63) (road l68 l50) (road l68 l57) (road l69 l28) (road l69 l44) (road l69 l72) (road l7 l80) (road l7 l9) (road l70 l22) (road l70 l26) (road l71 l30) (road l71 l34) (road l72 l0) (road l72 l69) (road l73 l17) (road l73 l56) (road l74 l42) (road l74 l62) (road l75 l37) (road l75 l40) (road l76 l32) (road l76 l56) (road l77 l46) (road l77 l55) (road l78 l12) (road l78 l29) (road l79 l14) (road l79 l36) (road l8 l41) (road l8 l6) (road l80 l0) (road l80 l7) (road l81 l10) (road l81 l17) (road l82 l34) (road l82 l53) (road l82 l55) (road l83 l18) (road l83 l9) (road l9 l7) (road l9 l83) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l42) (spare-in l43) (spare-in l45) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l73) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l82) (vehicle-at l46))
    (:goal (vehicle-at l21))
)